import enum
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    placeholder_token_id: int
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float
