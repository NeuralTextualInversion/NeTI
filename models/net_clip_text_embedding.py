from typing import Optional, Tuple

import torch
from torch import nn
from transformers import CLIPTextConfig

from models.neti_mapper import NeTIMapper
from utils.types import NeTIBatch


class NeTICLIPTextEmbeddings(nn.Module):
    """ Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def set_mapper(self, mapper: NeTIMapper):
        self.mapper = mapper

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                batch: Optional[NeTIBatch] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if batch is not None:
            input_ids = batch.input_ids

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        ####################################################################
        # NeTI logic - Use mapper to overwrite the learnable token embedding
        ####################################################################
        bypass_outputs = None
        if batch is not None:
            mapper_outputs = self.mapper(timestep=batch.timesteps.float(),
                                         unet_layer=batch.unet_layers.float(),
                                         truncation_idx=batch.truncation_idx)
            mapper_outputs = mapper_outputs.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            if self.mapper.output_bypass:
                bypass_outputs = mapper_outputs[:, mapper_outputs.shape[1] // 2:]
                mapper_outputs = mapper_outputs[:, :mapper_outputs.shape[1] // 2]

            # Overwrite the index of the placeholder token with the mapper output for each entry in the batch
            learnable_idxs = (input_ids == batch.placeholder_token_id).nonzero(as_tuple=True)[1]
            inputs_embeds[torch.arange(input_ids.shape[0]), learnable_idxs] = mapper_outputs

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings, bypass_outputs
