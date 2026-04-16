from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B3_Weights, ResNet18_Weights, efficientnet_b3, resnet18

from .position_encoding import build_position_encoding


class ImageBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, hidden_dim: int) -> None:
        super().__init__()
        self.position_embedding = build_position_encoding(hidden_dim)

        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.num_channels = 512
        elif name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b3(weights=weights)
            self.features = backbone.features
            self.num_channels = 1536
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, image: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = self.features(image)
        pos = self.position_embedding(features).to(features.dtype)
        return [features], [pos]


def build_backbone_local(name: str, pretrained: bool, hidden_dim: int) -> ImageBackbone:
    return ImageBackbone(name=name, pretrained=pretrained, hidden_dim=hidden_dim)
