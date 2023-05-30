from typing import Union

import torch
from torch import nn


class NeTIPositionalEncoding(nn.Module):

    def __init__(self, sigma_t: float, sigma_l: float, num_w: int = 1024):
        super().__init__()
        self.sigma_t = sigma_t
        self.sigma_l = sigma_l
        self.num_w = num_w
        self.w = torch.randn((num_w, 2))
        self.w[:, 0] *= sigma_t
        self.w[:, 1] *= sigma_l
        self.w = nn.Parameter(self.w).cuda()

    def encode(self, t: Union[int, torch.Tensor], l: Union[int, torch.Tensor]):
        """ Maps the given time and layer input into a 2048-dimensional vector. """
        if type(t) == int or t.ndim == 0:
            x = torch.tensor([t, l]).float()
        else:
            x = torch.stack([t, l], dim=1).T
        x = x.cuda()
        v = torch.cat([torch.sin(self.w.detach() @ x), torch.cos(self.w.detach() @ x)])
        if type(t) == int:
            v_norm = v / v.norm()
        else:
            v_norm = v / v.norm(dim=0)
            v_norm = v_norm.T
        return v_norm

    def init_layer(self, num_time_anchors: int, num_layers: int) -> torch.Tensor:
        """ Computes the weights for the positional encoding layer of size 160x2048."""
        anchor_vectors = []
        for t_anchor in range(0, 1000, 1000 // num_time_anchors):
            for l_anchor in range(0, num_layers):
                anchor_vectors.append(self.encode(t_anchor, l_anchor).float())
        A = torch.stack(anchor_vectors)
        return A


class BasicEncoder(nn.Module):
    """ Simply normalizes the given timestep and unet layer to be between -1 and 1. """

    def __init__(self, num_denoising_timesteps: int = 1000, num_unet_layers: int = 16):
        super().__init__()
        self.normalized_timesteps = (torch.arange(num_denoising_timesteps) / (num_denoising_timesteps - 1)) * 2 - 1
        self.normalized_unet_layers = (torch.arange(num_unet_layers) / (num_unet_layers - 1)) * 2 - 1
        self.normalized_timesteps = nn.Parameter(self.normalized_timesteps).cuda()
        self.normalized_unet_layers = nn.Parameter(self.normalized_unet_layers).cuda()

    def encode(self, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        normalized_input = torch.stack([self.normalized_timesteps[timestep.long()],
                                        self.normalized_unet_layers[unet_layer.long()]]).T
        return normalized_input
