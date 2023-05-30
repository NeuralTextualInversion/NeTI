from pathlib import Path
from typing import Tuple

import pyrallis
import torch
from accelerate import Accelerator
from torch import nn
from transformers import CLIPTokenizer

from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder
from training.config import RunConfig


class CheckpointHandler:

    def __init__(self, cfg: RunConfig, placeholder_token_string: str, placeholder_token_id: int, save_root: Path):
        self.cfg = cfg
        self.placeholder_token_string = placeholder_token_string
        self.placeholder_token_id = placeholder_token_id
        self.save_root = save_root

    def save_model(self, text_encoder: NeTICLIPTextModel,
                   accelerator: Accelerator,
                   embeds_save_name: str,
                   mapper_save_name: str):
        self.save_learned_embeds(text_encoder, accelerator, embeds_save_name)
        self.save_mapper(text_encoder, mapper_save_name)

    def save_learned_embeds(self, text_encoder: NeTICLIPTextModel, accelerator: Accelerator, save_name: str):
        """
        Save learned embeddings. This embedding isn't really learned, but we'll add it to the tokenizer at inference
        to take the place of our placeholder token.
        """
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[self.placeholder_token_id]
        learned_embeds = learned_embeds.detach().cpu()
        learned_embeds_dict = {self.placeholder_token_string: learned_embeds}
        torch.save(learned_embeds_dict, self.save_root / save_name)

    def save_mapper(self, text_encoder: NeTICLIPTextModel, save_name: str):
        """ Save the mapper and config to be used at inference. """
        cfg_ = RunConfig(**self.cfg.__dict__.copy())
        state_dict = {
            "state_dict": text_encoder.text_model.embeddings.mapper.state_dict(),
            "cfg": pyrallis.encode(cfg_),
            "encoder": text_encoder.text_model.embeddings.mapper.encoder
        }
        torch.save(state_dict, self.save_root / save_name)

    @staticmethod
    def load_mapper(mapper_path: Path) -> Tuple[RunConfig, NeTIMapper]:
        mapper_ckpt = torch.load(mapper_path, map_location="cpu")
        cfg = pyrallis.decode(RunConfig, mapper_ckpt['cfg'])
        neti_mapper = NeTIMapper(output_dim=768,
                                 use_nested_dropout=cfg.model.use_nested_dropout,
                                 nested_dropout_prob=cfg.model.nested_dropout_prob,
                                 norm_scale=cfg.model.target_norm,
                                 use_positional_encoding=cfg.model.use_positional_encoding,
                                 num_pe_time_anchors=cfg.model.num_pe_time_anchors,
                                 pe_sigmas=cfg.model.pe_sigmas,
                                 output_bypass=cfg.model.output_bypass)
        neti_mapper.load_state_dict(mapper_ckpt['state_dict'], strict=True)
        encoder = mapper_ckpt['encoder']
        if isinstance(encoder, NeTIPositionalEncoding):
            encoder.w = nn.Parameter(mapper_ckpt['encoder'].w.cuda())
        elif isinstance(encoder, BasicEncoder):
            encoder.normalized_timesteps = mapper_ckpt['encoder'].normalized_timesteps.cuda()
            encoder.normalized_unet_layers = mapper_ckpt['encoder'].normalized_unet_layers.cuda()
        neti_mapper.encoder = encoder.cuda()
        neti_mapper.cuda()
        neti_mapper.eval()
        return cfg, neti_mapper

    @staticmethod
    def load_learned_embed_in_clip(learned_embeds_path: Path,
                                   text_encoder: NeTICLIPTextModel,
                                   tokenizer: CLIPTokenizer) -> Tuple[str, int]:
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_tokens = list(loaded_learned_embeds.keys())
        embeds = list(loaded_learned_embeds.values())

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        embeds = [e.to(dtype) for e in embeds]

        # add the tokens in tokenizer
        num_added_tokens = tokenizer.add_tokens(trained_tokens)
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {trained_tokens[0]}. "
                             f"Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        placeholder_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in trained_tokens]

        for idx, (token, token_id, embed) in enumerate(zip(trained_tokens, placeholder_token_ids, embeds)):
            text_encoder.get_input_embeddings().weight.data[token_id] = embed

        assert len(trained_tokens) == 1, "Only one placeholder token is supported"
        placeholder_token = trained_tokens[0]
        placeholder_token_id = placeholder_token_ids[0]
        return placeholder_token, placeholder_token_id
