from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextModel, CLIPEncoder
from transformers.models.clip.modeling_clip import CLIPTextTransformer, _expand_mask

from models.net_clip_text_embedding import NeTICLIPTextEmbeddings
from utils.types import NeTIBatch


class NeTICLIPTextModel(CLIPTextModel):
    """ Modification of CLIPTextModel to use our NeTI mapper for computing the embeddings of the concept. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = NeTICLIPTextTransformer(config)
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                batch: Optional[NeTIBatch] = None) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.text_model.forward(
            batch=batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class NeTICLIPTextTransformer(CLIPTextTransformer):
    """ Modification of CLIPTextTransformer to use our NeTI mapper for computing the embeddings of the concept. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config=config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = NeTICLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                batch: Optional[NeTIBatch] = None) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bypass_output = None

        if input_ids is not None:  # Regular embedding logic
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            hidden_states, _ = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        ###########################
        # NeTI logic
        ###########################
        elif batch is not None:
            input_shape = batch.input_ids.size()
            batch.input_ids = batch.input_ids.view(-1, input_shape[-1])
            hidden_states, bypass_output = self.embeddings(batch=batch, position_ids=position_ids)

        else:
            raise ValueError("You have to specify either batch or input_ids!")

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state_with_bypass = last_hidden_state.clone()

        ###############################################
        # NeTI logic - compute the scaled bypass output
        ###############################################
        if bypass_output is not None:
            learnable_idxs = (batch.input_ids == batch.placeholder_token_id).nonzero(as_tuple=True)[1]
            existing_state = last_hidden_state_with_bypass[torch.arange(last_hidden_state.shape[0]), learnable_idxs]
            bypass_output = bypass_output / bypass_output.norm(dim=1, keepdim=True) \
                            * existing_state.norm(dim=1, keepdim=True)
            new_state = existing_state + 0.2 * bypass_output
            new_state = new_state.to(dtype=hidden_states.dtype)
            last_hidden_state_with_bypass[torch.arange(last_hidden_state.shape[0]), learnable_idxs] = new_state

        last_hidden_state = self.final_layer_norm(last_hidden_state)
        last_hidden_state_with_bypass = self.final_layer_norm(last_hidden_state_with_bypass)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        if input_ids is not None:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0]), input_ids.to(torch.int).argmax(dim=-1)
            ]
            pooled_output_with_bypass = last_hidden_state_with_bypass[
                torch.arange(last_hidden_state_with_bypass.shape[0]), input_ids.to(torch.int).argmax(dim=-1)
            ]
        elif batch is not None:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0]), batch.input_ids.to(torch.int).argmax(dim=-1)
            ]
            pooled_output_with_bypass = last_hidden_state_with_bypass[
                torch.arange(last_hidden_state_with_bypass.shape[0]), batch.input_ids.to(torch.int).argmax(dim=-1)
            ]
        else:
            raise ValueError("You have to specify either batch or input_ids!")

        if bypass_output is not None:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            ), BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state_with_bypass,
                pooler_output=pooled_output_with_bypass,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            ), None
