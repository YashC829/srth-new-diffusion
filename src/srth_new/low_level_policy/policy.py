from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from .models import build_act_model


class ACTPolicy(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.config = config
        self.model = build_act_model(config)
        self.kl_weight = float(config["kl_weight"])
        self.num_queries = self.model.num_queries
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        image = self.normalize(image)

        if actions is not None:
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            action_hat, _, (mu, logvar) = self.model(
                qpos=qpos,
                image=image,
                actions=actions,
                is_pad=is_pad,
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            l1 = (F.l1_loss(actions, action_hat, reduction="none") * ~is_pad.unsqueeze(-1)).mean()
            return {
                "l1": l1,
                "kl": total_kld[0],
                "loss": l1 + total_kld[0] * self.kl_weight,
            }

        action_hat, _, _ = self.model(qpos=qpos, image=image)
        return action_hat

    def configure_optimizers(self) -> torch.optim.Optimizer:
        backbone_params = []
        other_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "backbones" in name:
                backbone_params.append(parameter)
            else:
                other_params.append(parameter)

        return torch.optim.AdamW(
            [
                {"params": other_params},
                {
                    "params": backbone_params,
                    "lr": float(self.config["lr_backbone"]),
                },
            ],
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    def serialize(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

    def deserialize(self, model_state_dict: dict[str, torch.Tensor]):
        return self.load_state_dict(model_state_dict)


def kl_divergence(
    mu: torch.Tensor | None, logvar: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mu is None or logvar is None:
        zero = torch.zeros(1)
        return zero, zero, zero

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, keepdim=True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, keepdim=True)
    return total_kld, dimension_wise_kld, mean_kld
