from __future__ import annotations

import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int, temperature: int = 10000, normalize: bool = True) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        not_mask = torch.ones_like(tensor[:, :1])
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, 0, :, :, None] / dim_t
        pos_y = y_embed[:, 0, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def build_position_encoding(hidden_dim: int) -> nn.Module:
    return PositionEmbeddingSine(hidden_dim // 2)
