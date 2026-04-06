from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from srth_new.low_level_policy.models.detr.main import build_ACT_model_and_optimizer
from .backbone import build_backbone
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer

import logging
log = logging.getLogger(__name__)

def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)
    return mu + eps * std


def get_sinusoid_encoding_table(num_positions: int, hidden_dim: int) -> torch.Tensor:
    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_idx // 2) / hidden_dim)
            for hid_idx in range(hidden_dim)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos) for pos in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float().unsqueeze(0)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DETRVAE(nn.Module):
    def __init__(
        self,
        backbones: list[nn.Module],
        transformer: nn.Module,
        encoder: nn.Module,
        state_dim: int,
        action_dim: int,
        num_queries: int,
        camera_names: list[str],
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.hidden_dim = transformer.d_model
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.backbones = nn.ModuleList(backbones)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(state_dim, self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, self.hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, self.hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(2 + num_queries, self.hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, self.hidden_dim)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        is_training = actions is not None
        batch_size = qpos.shape[0]

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1).permute(1, 0, 2)

            cls_joint_is_pad = torch.zeros((batch_size, 2), dtype=torch.bool, device=qpos.device)
            encoder_is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)
            pos_embed = self.pos_table[:, : encoder_input.shape[0]].detach().clone().permute(1, 0, 2)
            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=encoder_is_pad,
            )
            latent_info = self.latent_proj(encoder_output[0])
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = None
            logvar = None
            latent_sample = torch.zeros((batch_size, self.latent_dim), device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)
        proprio_input = self.input_proj_robot_state(qpos)

        all_cam_features = []
        all_cam_pos = []
        for cam_idx in range(len(self.camera_names)):
            features, pos = self.backbones[cam_idx](image[:, cam_idx])
            all_cam_features.append(self.input_proj(features[0]))
            all_cam_pos.append(pos[0])

        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)
        hs = self.transformer(
            src=src,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=pos,
            latent_input=latent_input,
            proprio_input=proprio_input,
            additional_pos_embed=self.additional_pos_embed.weight,
        )[0]

        action_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return action_hat, is_pad_hat, (mu, logvar)


def build_encoder(config: dict[str, object]) -> nn.Module:
    encoder_layer = TransformerEncoderLayer(
        d_model=int(config["hidden_dim"]),
        nhead=int(config["nheads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
    )
    return TransformerEncoder(encoder_layer, int(config["enc_layers"]))


def build_act_model(config: dict[str, object]) -> DETRVAE:
    backbones = [
        build_backbone(
            name=str(config["backbone"]),
            pretrained=bool(config["pretrained_backbone"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        for _ in config["camera_names"]
    ]

    transformer = build_transformer(config)
    encoder = build_encoder(config)
    return DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        num_queries=int(config["num_queries"]),
        camera_names=list(config["camera_names"]),
    )

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        log.info(f"KL Weight {self.kl_weight}")
        multi_gpu = args_override["multi_gpu"]
        self.num_queries = (
            self.model.module.num_queries if multi_gpu else self.model.num_queries
        )

    def __call__(self, qpos, image, actions=None, is_pad=None, command_embedding=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                command_embedding=command_embedding,
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, command_embedding=command_embedding
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)