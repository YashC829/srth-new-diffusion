from __future__ import annotations

from torch.autograd import Variable

import numpy as np
import torch
from torch import nn

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class DETRVAE_Decoder(nn.Module):
    """This is the decoder only transformer"""

    def __init__(
        self,
        backbones,
        transformer_decoder,
        state_dim,
        num_queries,
        camera_names,
        action_dim,
        use_language=False,
        use_film=False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.cam_num = len(camera_names)
        self.transformer_decoder = transformer_decoder
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer_decoder.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # self.proprio_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.use_language = use_language
        self.use_film = use_film
        if use_language:
            self.lang_embed_proj = nn.Linear(
                768, hidden_dim
            )  # 512 / 768 for clip / distilbert

        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            # self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            # self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            # self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        # encoder extra parameters
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq
        self.additional_pos_embed = nn.Embedding(
            1, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, image, actions=None, is_pad=None, command_embedding=None):

        # Project the command embedding to the required dimension
        if command_embedding is not None:
            if self.use_language:
                command_embedding_proj = self.lang_embed_proj(command_embedding)
            else:
                raise NotImplementedError
        # if self.feature_loss:
        #     # bs,_,_,h,w = image.shape
        #     image_future = image[:,len(self.camera_names):].clone()
        #     image = image[:,:len(self.camera_names)].clone()

        if len(self.backbones) > 1:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                if self.use_film:
                    # features, pos = self.backbones[0](image[:, cam_id], command_embedding)
                    features, pos = self.backbones[cam_id](
                        image[:, cam_id], command_embedding
                    )
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
        else:
            all_cam_features = []
            all_cam_pos = []
            bs = image.shape[0]
            features, pos = self.backbones[0](
                image.reshape([-1, 3, image.shape[-2], image.shape[-1]]),
                command_embedding,
            )
            project_feature = self.input_proj(features[0])
            project_feature = project_feature.reshape(
                [
                    bs,
                    self.cam_num,
                    project_feature.shape[1],
                    project_feature.shape[2],
                    project_feature.shape[3],
                ]
            )
            for i in range(self.cam_num):
                all_cam_features.append(project_feature[:, i, :])
                all_cam_pos.append(pos[0])
        # proprioception features
        # proprio_input = self.input_proj_robot_state(qpos) #B, 512
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)  # B, 512,12,26
        pos = torch.cat(all_cam_pos, axis=3)  # B, 512,12,26
        # Only append the command embedding if we are using one-hot
        command_embedding_to_append = (
            command_embedding_proj if self.use_language else None
        )

        hs = self.transformer_decoder(
            src,
            self.query_embed.weight,
            pos_embed=pos,
            additional_pos_embed=self.additional_pos_embed.weight,
            command_embedding=command_embedding_to_append,
        )  # B, chunk_size, 512

        # Print the shape of hs
        # print("Shape of hs:", hs.shape)

        # a_hat = self.action_head(hs) #B, chunk_size, action_dim
        hs_action = hs[:, -1 * self.num_queries :, :].clone()  # B, action_dim, 512
        # hs_img = hs[:,1:-1*self.num_queries,:].clone() #B, image_feature_dim, 512 #final image feature
        # hs_proprio = hs[:,[0],:].clone() #B, proprio_feature_dim, 512
        a_hat = self.action_head(hs_action)
        # a_proprio = self.proprio_head(hs_proprio) #proprio head
        # if self.feature_loss and self.training:
        #     # proprioception features
        #     src_future = torch.cat(all_cam_features_future, axis=3) #B, 512,12,26
        #     src_future = src_future.flatten(2).permute(2, 0, 1).transpose(1, 0) # B, 12*26, 512
        #     hs_img = {'hs_img': hs_img, 'src_future': src_future}

        return a_hat

class DETRVAE(nn.Module):
    """DETRVAE variant with optional depth and optional history conditioning."""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        depth_backbone=None,
        history_chunk_size=None,
        history_num_tokens=8,
        history_num_layers=2,
        history_num_heads=8,
        use_language=False,
        use_film=False,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.input_size = state_dim
        self.hidden_dim = hidden_dim = transformer.d_model
        self.use_language = use_language
        self.use_film = use_film
        self.use_depth = depth_backbone is not None
        self.use_history = history_chunk_size is not None and history_chunk_size > 0
        self.history_num_tokens = history_num_tokens

        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if use_language:
            self.lang_embed_proj = nn.Linear(768, hidden_dim)

        if backbones is None:
            raise Exception("Must pass backbones to model. Backbones cannot be None")

        self.input_proj = nn.Conv2d(
            backbones[0].num_channels, hidden_dim, kernel_size=1
        )
        self.backbones = nn.ModuleList(backbones)

        if self.use_depth:
            self.depth_backbone = depth_backbone
            self.depth_1d_to_3d_proj = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=1
            )
            assert depth_backbone is not None
            self.depth_input_proj = nn.Conv2d(
                depth_backbone.num_channels, hidden_dim, kernel_size=1
            )
            self.aligned_modality_embed = nn.Embedding(2, hidden_dim)
            self.aligned_feature_fusion = nn.Conv2d(
                hidden_dim * 2, hidden_dim, kernel_size=1
            )
        else:
            self.depth_backbone = None

        self.input_proj_robot_state = nn.Linear(self.input_size, hidden_dim)

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(self.input_size, hidden_dim)
        self.encoder_joint_proj = nn.Linear(self.input_size, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)

        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        pos_embed_dim = 2
        if self.use_language:
            pos_embed_dim += 1
        if self.use_history:
            pos_embed_dim += history_num_tokens
        self.additional_pos_embed = nn.Embedding(pos_embed_dim, hidden_dim)

        if self.use_history:
            self.history_input_proj = nn.Linear(state_dim, hidden_dim)
            self.history_cls_tokens = nn.Embedding(history_num_tokens, hidden_dim)

            history_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=history_num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.history_encoder = nn.TransformerEncoder(
                history_layer,
                num_layers=history_num_layers,
            )

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
            [
                rgb_feature + rgb_modality,
                depth_feature + depth_modality,
            ],
            dim=1,
        )
        return self.aligned_feature_fusion(aligned_feature)

    def _encode_history(self, history, history_is_pad=None):
        """Encode history into a small set of conditioning tokens.

        Args:
            history: `(B, T_hist, state_dim)`
            history_is_pad: optional `(B, T_hist)` bool mask
        Returns:
            history_tokens: `(B, history_num_tokens, hidden_dim)`
        """
        if history is None:
            return None

        if not self.use_history:
            raise ValueError(
                "history was passed, but history_chunk_size was not provided in __init__."
            )

        bs = history.shape[0]

        history_embed = self.history_input_proj(history)
        history_queries = self.history_cls_tokens.weight[None].repeat(bs, 1, 1)

        encoder_input = torch.cat([history_queries, history_embed], dim=1)

        if history_is_pad is not None:
            query_pad = torch.zeros(
                bs,
                self.history_num_tokens,
                dtype=torch.bool,
                device=history.device,
            )
            history_is_pad = torch.cat([query_pad, history_is_pad], dim=1)

        encoded = self.history_encoder(
            encoder_input,
            src_key_padding_mask=history_is_pad,
        )

        return encoded[:, : self.history_num_tokens]

    def forward(
        self,
        qpos,
        image_stack,
        env_state=None,
        actions=None,
        is_pad=None,
        command_embedding=None,
        depth_image=None,
        history=None,
        history_is_pad=None,
    ):
        """Forward pass.

        Args:
            qpos: `(B, state_dim)`
            image_stack: `(B, num_rgb_cameras, C, H, W)`
            depth_image: optional `(B, 1, H, W)`, aligned with `image_stack[:, 0]`
            history: optional `(B, T_hist, state_dim)`
            history_is_pad: optional `(B, T_hist)`
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        command_embedding_proj = None

        if command_embedding is not None:
            if self.use_language:
                command_embedding_proj = self.lang_embed_proj(command_embedding)
            else:
                raise NotImplementedError

        history_tokens = self._encode_history(history, history_is_pad)

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)

            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed],
                dim=1,
            )
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full(
                (bs, 2),
                False,
                dtype=torch.bool,
                device=qpos.device,
            )
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)

            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad,
            )

            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)

            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]

            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                [bs, self.latent_dim],
                dtype=torch.float32,
                device=qpos.device,
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []

            first_rgb_feature = None
            first_rgb_pos = None

            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self._forward_backbone(
                    self.backbones[cam_id],
                    image_stack[:, cam_id],
                    command_embedding,
                )

                projected_feature = self.input_proj(features)

                if cam_id == 0:
                    first_rgb_feature = projected_feature
                    first_rgb_pos = pos
                else:
                    all_cam_features.append(projected_feature)
                    all_cam_pos.append(pos)

            if first_rgb_feature is None or first_rgb_pos is None:
                raise RuntimeError("Expected at least one RGB camera backbone.")

            if depth_image is not None:
                if not self.use_depth:
                    raise ValueError(
                        "depth_image was passed, but depth_backbone was not provided "
                        "in __init__."
                    )

                depth_3d = self.depth_1d_to_3d_proj(depth_image)

                depth_features, _ = self._forward_backbone(
                    self.depth_backbone,
                    depth_3d,
                    command_embedding,
                )

                depth_feature = self.depth_input_proj(depth_features)

                if depth_feature.shape[-2:] != first_rgb_feature.shape[-2:]:
                    raise ValueError(
                        "Depth and first RGB feature maps must share spatial shape "
                        "for aligned fusion."
                    )

                first_feature = self._fuse_aligned_rgbd(
                    first_rgb_feature,
                    depth_feature,
                )
            else:
                first_feature = first_rgb_feature

            all_cam_features.insert(0, first_feature)
            all_cam_pos.insert(0, first_rgb_pos)

            proprio_input = self.input_proj_robot_state(qpos)

            src = torch.cat(all_cam_features, dim=3)
            pos = torch.cat(all_cam_pos, dim=3)

            extra_conditioning_tokens = []

            if self.use_language and command_embedding_proj is not None:
                if command_embedding_proj.ndim == 2:
                    command_embedding_proj = command_embedding_proj.unsqueeze(1)
                extra_conditioning_tokens.append(command_embedding_proj)

            if history_tokens is not None:
                extra_conditioning_tokens.append(history_tokens)

            command_embedding_to_append = None
            if len(extra_conditioning_tokens) > 0:
                command_embedding_to_append = torch.cat(
                    extra_conditioning_tokens,
                    dim=1,
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
            transformer_input = torch.cat([qpos, env_state], dim=1)

            hs = self.transformer(
                transformer_input,
                None,
                self.query_embed.weight,
                self.pos.weight,
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]
