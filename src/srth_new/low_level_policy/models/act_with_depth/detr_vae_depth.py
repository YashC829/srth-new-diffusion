from __future__ import annotations

import torch
from torch import nn

from srth_new.low_level_policy.models.detr.models.detr_vae import (
    get_sinusoid_encoding_table,
    reparametrize,
)


class DETRVAEDepth(nn.Module):
    """Depth-aware DETRVAE variant.

    This class stays close to the original DETRVAE implementation and only
    changes how image features are built:

    - RGB cameras listed in `camera_names` are encoded exactly as before.
    - A separate depth image is encoded with its own backbone.
    - The depth feature map is fused with camera 0 at matching spatial
      locations before multi-camera concatenation so alignment is explicit.
    """

    def __init__(
        self,
        backbones,
        depth_backbone,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        use_language=False,
        use_film=False,
        num_command=2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.input_size = state_dim
        self.hidden_dim = hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.use_language = use_language
        self.use_film = use_film
        if use_language:
            self.lang_embed_proj = nn.Linear(768, hidden_dim)

        if backbones is None:
            raise Exception("Must pass backbones to model. Backbones cannot be None")

        self.input_proj = nn.Conv2d(
            backbones[0].num_channels, hidden_dim, kernel_size=1
        )
        self.depth_1d_to_3d_proj = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.depth_input_proj = nn.Conv2d(
            depth_backbone.num_channels, hidden_dim, kernel_size=1
        )
        self.backbones = nn.ModuleList(backbones)
        self.depth_backbone = depth_backbone
        self.input_proj_robot_state = nn.Linear(self.input_size, hidden_dim)
        self.aligned_modality_embed = nn.Embedding(2, hidden_dim)
        self.aligned_feature_fusion = nn.Conv2d(
            hidden_dim * 2, hidden_dim, kernel_size=1
        )

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(self.input_size, hidden_dim)
        self.encoder_joint_proj = nn.Linear(self.input_size, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        pos_embed_dim = 3 if self.use_language else 2
        self.additional_pos_embed = nn.Embedding(pos_embed_dim, hidden_dim)

    def _forward_backbone(self, backbone, image, command_embedding):
        if self.use_film:
            features, pos = backbone(image, command_embedding)
        else:
            features, pos = backbone(image)
        return features[0], pos[0]

    def _fuse_aligned_rgbd(self, rgb_feature, depth_feature):
        rgb_modality = self.aligned_modality_embed.weight[0][None, :, None, None]
        depth_modality = self.aligned_modality_embed.weight[1][None, :, None, None]
        aligned_feature = torch.cat(
            [rgb_feature + rgb_modality, depth_feature + depth_modality], dim=1
        )
        return self.aligned_feature_fusion(aligned_feature)

    def forward(
        self,
        qpos,
        image_stack,
        depth_image,
        env_state,
        actions=None,
        is_pad=None,
        command_embedding=None
    ):
        """Forward pass.

        Args:
            qpos: `(B, state_dim)`
            image: `(B, num_rgb_cameras, C, H, W)`
            depth_image: `(B, C, H, W)`, aligned with `image[:, 0]`
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        command_embedding_proj = None

        if command_embedding is not None:
            if self.use_language:
                command_embedding_proj = self.lang_embed_proj(command_embedding)
            else:
                raise NotImplementedError

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            if depth_image is None:
                raise ValueError(
                    "depth_image is required for DETRVAEDepth and must align with "
                    "the first RGB camera."
                )

            all_cam_features = []
            all_cam_pos = []

            first_rgb_feature = None
            first_rgb_pos = None
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self._forward_backbone(
                    self.backbones[cam_id], image_stack[:, cam_id], command_embedding
                )
                projected_feature = self.input_proj(features)
                if cam_id == 0:
                    first_rgb_feature = projected_feature
                    first_rgb_pos = pos
                else:
                    all_cam_features.append(projected_feature)
                    all_cam_pos.append(pos)

            depth_3d = self.depth_1d_to_3d_proj(depth_image)
            depth_features, _ = self._forward_backbone(
                self.depth_backbone, depth_3d, command_embedding
            )
            depth_feature = self.depth_input_proj(depth_features)

            if first_rgb_feature is None or first_rgb_pos is None:
                raise RuntimeError("Expected at least one RGB camera backbone.")
            if depth_feature.shape[-2:] != first_rgb_feature.shape[-2:]:
                raise ValueError(
                    "Depth and first RGB feature maps must share spatial shape for "
                    "aligned fusion."
                )

            fused_first_feature = self._fuse_aligned_rgbd(
                first_rgb_feature, depth_feature
            )
            all_cam_features.insert(0, fused_first_feature)
            all_cam_pos.insert(0, first_rgb_pos)

            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            command_embedding_to_append = (
                command_embedding_proj if self.use_language else None
            )

            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                command_embedding=command_embedding_to_append,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
