from typing import Dict, Optional

import torch
from diffusers.models.cross_attention import CrossAttention


class XTIAttenProc:

    def __call__(self, attn: CrossAttention,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[Dict[str, torch.Tensor]] = None,
                 attention_mask: Optional[torch.Tensor] = None):

        _ehs_bypass = None
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, dict):
                this_idx = encoder_hidden_states["this_idx"]
                _ehs = encoder_hidden_states[f"CONTEXT_TENSOR_{this_idx}"]
                if f"CONTEXT_TENSOR_BYPASS_{this_idx}" in encoder_hidden_states:
                    _ehs_bypass = encoder_hidden_states[f"CONTEXT_TENSOR_BYPASS_{this_idx}"]
                encoder_hidden_states["this_idx"] += 1
                encoder_hidden_states["this_idx"] %= 16
            else:
                _ehs = encoder_hidden_states
        else:
            _ehs = None

        batch_size, sequence_length, _ = (hidden_states.shape if _ehs is None else _ehs.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if _ehs is None:
            _ehs = hidden_states
        elif attn.cross_attention_norm:
            _ehs = attn.norm_cross(_ehs)
            _ehs_bypass = attn.norm_cross(_ehs_bypass)

        key = attn.to_k(_ehs)
        if _ehs_bypass is not None:
            value = attn.to_v(_ehs_bypass)
        else:
            value = attn.to_v(_ehs)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
