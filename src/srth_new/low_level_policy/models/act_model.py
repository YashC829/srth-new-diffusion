from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from srth_new.general.utils import processing
from srth_new.general.utils.lang_encoding import encode_text, initialize_model_and_tokenizer
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
    """ACT policy wrapper used by the low-level training and inference code.

    This class layers project-specific behavior around the DETR/ACT backbone:

    - builds the ACT model and optimizer from the DETR configuration
    - caches and applies dataset statistics used to normalize action targets
    - converts between absolute robot actions and the relative policy action
      representation expected by the network
    - optionally encodes language commands for conditioned policies
    - serializes both learned weights and lightweight metadata needed to resume
      training or run inference later

    A subtle but important detail is that the internal ACT model currently
    receives a zero-valued proprioceptive input (`model_qpos`) during the
    forward pass. The externally supplied `current_pose` is still required,
    because it is used to transform dataset actions into the policy's relative
    action space during training and to convert predicted policy actions back
    into absolute robot commands during inference.
    """

    def __init__(self, args_override):
        """Initialize the policy, optimizer, and optional language encoder.

        Args:
            args_override: Mapping of ACT / DETR configuration values. The
                contents are forwarded to `build_ACT_model_and_optimizer`, then
                a few policy-specific keys such as `kl_weight`,
                `action_mode`, `norm_scheme`, and language settings are read
                directly from the same mapping.
        """
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.action_dim = int(args_override["action_dim"])
        self.state_dim = int(args_override["action_dim"])
        self.action_mode = str(args_override.get("action_mode", "hybrid_relative"))
        self.norm_scheme = str(args_override.get("norm_scheme", "std"))
        self.use_language = bool(args_override.get("use_language", False))
        self.language_encoder = str(args_override.get("language_encoder", "distilbert"))
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self._command_embedding_cache = {}
        self.tokenizer = None
        self.language_model = None
        if self.use_language:
            self.tokenizer, self.language_model = initialize_model_and_tokenizer(
                self.language_encoder
            )
            self.language_model.eval()

        self.register_buffer("stats_mean", torch.zeros(self.action_dim))
        self.register_buffer("stats_std", torch.ones(self.action_dim))
        self.register_buffer("stats_min", torch.zeros(self.action_dim))
        self.register_buffer("stats_max", torch.zeros(self.action_dim))
        self.register_buffer("has_dataset_stats", torch.tensor(False))
        self.stats_dataset_dir = None
        self.stats_tissue_sample_ids_train = None
        log.info(f"KL Weight {self.kl_weight}")
        self.num_queries = self.model.num_queries

    def set_dataset_stats(self, dataset_stats: processing.DatasetStats) -> None:
        """Attach dataset statistics used for action normalization.

        The stored statistics are saved as buffers so they travel with the
        module state dict and are restored automatically from checkpoints.

        Args:
            dataset_stats: Precomputed action statistics for the training split.
                The policy expects mean/std/min/max arrays shaped like the
                action vector.
        """
        self.stats_mean.copy_(torch.as_tensor(dataset_stats.mean, dtype=torch.float32))
        self.stats_std.copy_(torch.as_tensor(dataset_stats.std, dtype=torch.float32))
        self.stats_min.copy_(torch.as_tensor(dataset_stats.min, dtype=torch.float32))
        self.stats_max.copy_(torch.as_tensor(dataset_stats.max, dtype=torch.float32))
        self.has_dataset_stats.fill_(True)
        self.stats_dataset_dir = dataset_stats.dataset_dir
        self.stats_tissue_sample_ids_train = list(dataset_stats.tissue_sample_ids_train)

    def export_dataset_stats(self) -> processing.DatasetStats:
        """Return the currently attached dataset statistics as a dataclass.

        Returns:
            A `DatasetStats` object containing normalization arrays and the
            dataset metadata associated with them.

        Raises:
            RuntimeError: If statistics have not yet been attached to the
                policy.
        """
        if not bool(self.has_dataset_stats.item()):
            raise RuntimeError("Dataset statistics have not been set on the policy.")

        return processing.DatasetStats(
            mean=self.stats_mean.detach().cpu().numpy().copy(),
            std=self.stats_std.detach().cpu().numpy().copy(),
            min=self.stats_min.detach().cpu().numpy().copy(),
            max=self.stats_max.detach().cpu().numpy().copy(),
            dataset_dir=self.stats_dataset_dir,
            tissue_sample_ids_train=list(self.stats_tissue_sample_ids_train or []),
        )

    def _encode_command_text(self, command_text, device: torch.device):
        """Encode one or more command strings for language-conditioned ACT.

        Encoded text embeddings are cached by string value to avoid repeated
        encoder calls during training when the same command appears many times.

        Args:
            command_text: A single string or a batch of strings. Must be
                provided when `use_language=True`.
            device: Device where the returned embedding tensor should live.

        Returns:
            A tensor of shape `(batch, embedding_dim)` or `None` when language
            conditioning is disabled.
        """
        if not self.use_language:
            return None
        if command_text is None:
            raise ValueError("command_text is required when use_language=True")

        if isinstance(command_text, str):
            texts = [command_text]
        else:
            texts = list(command_text)

        embeddings = []
        for text in texts:
            if text not in self._command_embedding_cache:
                embedding = torch.as_tensor(
                    encode_text(
                        text,
                        self.language_encoder,
                        self.tokenizer,
                        self.language_model,
                    ),
                    dtype=torch.float32,
                ).flatten()
                self._command_embedding_cache[text] = embedding.cpu()
            embeddings.append(self._command_embedding_cache[text])

        return torch.stack(embeddings, dim=0).to(device)

    def _require_dataset_stats(self) -> None:
        if not bool(self.has_dataset_stats.item()):
            raise RuntimeError(
                "Dataset statistics must be set on the policy before normalization."
            )

    @staticmethod
    def _preserve_rotation_columns(normalized: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        normalized[..., 3:9] = raw[..., 3:9]
        normalized[..., 13:19] = raw[..., 13:19]
        return normalized

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize policy-space actions with the configured statistics scheme.

        Positional and gripper dimensions are normalized, while the 6D rotation
        blocks are copied back from the raw tensor unchanged. This matches the
        representation expected by the current action-conversion pipeline.

        Args:
            actions: Tensor of policy-space actions.

        Returns:
            The normalized action tensor.
        """
        self._require_dataset_stats()
        stats_mean = self.stats_mean.to(device=actions.device, dtype=actions.dtype)
        stats_std = self.stats_std.to(device=actions.device, dtype=actions.dtype)
        stats_min = self.stats_min.to(device=actions.device, dtype=actions.dtype)
        stats_max = self.stats_max.to(device=actions.device, dtype=actions.dtype)

        if self.norm_scheme == "std":
            normalized = (actions - stats_mean) / stats_std
        elif self.norm_scheme == "min_max":
            normalized = (actions - stats_min) / (stats_max - stats_min) * 2 - 1
        else:
            raise NotImplementedError(f"Unsupported norm scheme: {self.norm_scheme}")

        return self._preserve_rotation_columns(normalized, actions)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert `normalize_actions` using the stored dataset statistics.

        Args:
            actions: Normalized policy-space action tensor.

        Returns:
            The denormalized action tensor in the policy action space.
        """
        self._require_dataset_stats()
        stats_mean = self.stats_mean.to(device=actions.device, dtype=actions.dtype)
        stats_std = self.stats_std.to(device=actions.device, dtype=actions.dtype)
        stats_min = self.stats_min.to(device=actions.device, dtype=actions.dtype)
        stats_max = self.stats_max.to(device=actions.device, dtype=actions.dtype)

        if self.norm_scheme == "std":
            denormalized = actions * stats_std + stats_mean
        elif self.norm_scheme == "min_max":
            denormalized = (actions + 1) / 2 * (stats_max - stats_min) + stats_min
        else:
            raise NotImplementedError(f"Unsupported norm scheme: {self.norm_scheme}")

        return self._preserve_rotation_columns(denormalized, actions)

    def _convert_single_raw_action_sequence_to_policy_actions(
        self,
        current_pose: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        qpos_psm1 = current_pose[:8]
        qpos_psm2 = current_pose[8:16]
        actions_psm1 = actions[:, :8]
        actions_psm2 = actions[:, 8:16]

        if self.action_mode == "hybrid_relative":
            diff_psm1 = processing.computer_diff_actions(qpos_psm1, actions_psm1)
            diff_psm2 = processing.computer_diff_actions(qpos_psm2, actions_psm2)
        elif self.action_mode == "ego":
            diff_psm1 = processing.compute_relative_actions_in_SE3(qpos_psm1, actions_psm1)
            diff_psm2 = processing.compute_relative_actions_in_SE3(qpos_psm2, actions_psm2)
        elif self.action_mode == "relative_endoscope":
            diff_psm1 = processing.compute_diff_actions_relative_endoscope(qpos_psm1, actions_psm1)
            diff_psm2 = processing.compute_diff_actions_relative_endoscope(qpos_psm2, actions_psm2)
        else:
            raise NotImplementedError(f"Unsupported action mode: {self.action_mode}")

        return np.column_stack((diff_psm1, diff_psm2)).astype(np.float32)

    def prepare_actions_for_training(
        self,
        current_pose: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert absolute dataset actions into normalized policy targets.

        The dataset stores per-timestep absolute robot actions. ACT is trained
        on a relative policy representation, so each valid action in the chunk
        is converted relative to the provided `current_pose`, normalized using
        dataset statistics, and padded positions are zeroed out.

        Args:
            current_pose: Tensor shaped `(B, 16)` or `(16,)` describing the
                current left and right arm poses as
                `[xyz, quat_xyzw, jaw] * 2`.
            actions: Absolute future action sequence shaped `(B, T, 16)` or
                `(T, 16)`.
            is_pad: Padding mask for `actions`, where padded steps are `True`.
            device: Device for the returned tensor.

        Returns:
            A normalized policy-action tensor shaped like `actions`, but in the
            policy action space.
        """
        if current_pose.dim() == 1:
            current_pose = current_pose.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if is_pad.dim() == 1:
            is_pad = is_pad.unsqueeze(0)

        current_pose_np = current_pose.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        is_pad_np = is_pad.detach().cpu().numpy().astype(bool)

        policy_actions_np = np.zeros(
            (actions_np.shape[0], actions_np.shape[1], self.action_dim),
            dtype=np.float32,
        )
        for batch_idx, (pose, action, pad_mask) in enumerate(
            zip(current_pose_np, actions_np, is_pad_np)
        ):
            valid_mask = ~pad_mask
            if not np.any(valid_mask):
                continue

            valid_actions = action[valid_mask]
            converted_actions = self._convert_single_raw_action_sequence_to_policy_actions(
                pose, valid_actions
            )
            policy_actions_np[batch_idx, valid_mask] = converted_actions

        policy_actions = torch.from_numpy(policy_actions_np).to(device=device, dtype=torch.float32)
        policy_actions = self.normalize_actions(policy_actions)
        policy_actions = policy_actions.masked_fill(is_pad.unsqueeze(-1), 0.0)
        return policy_actions

    def _convert_single_policy_actions_to_absolute(
        self,
        current_pose: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        qpos_psm1 = current_pose[:8]
        qpos_psm2 = current_pose[8:16]
        chunk_size = actions.shape[0]

        if self.action_mode == "hybrid_relative":
            actions_psm1 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm1[:, 0:3] = qpos_psm1[0:3] + actions[:, 0:3]
            actions_psm1 = processing.convert_delta_6d_to_taskspace_quat(
                actions[:, 0:10], actions_psm1, qpos_psm1.copy()
            )
            actions_psm1[:, 7] = np.clip(actions[:, 9], -0.698, 0.698)

            actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm2[:, 0:3] = qpos_psm2[0:3] + actions[:, 10:13]
            actions_psm2 = processing.convert_delta_6d_to_taskspace_quat(
                actions[:, 10:20], actions_psm2, qpos_psm2.copy()
            )
            actions_psm2[:, 7] = np.clip(actions[:, 19], -0.698, 0.698)
        elif self.action_mode == "relative_endoscope":
            actions_psm1 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm1[:, 0:3] = qpos_psm1[0:3] + actions[:, 0:3]
            actions_psm1 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(
                actions[:, 0:10], actions_psm1, qpos_psm1.copy()
            )
            actions_psm1[:, 7] = np.clip(actions[:, 9], -0.698, 0.698)

            actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm2[:, 0:3] = qpos_psm2[0:3] + actions[:, 10:13]
            actions_psm2 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(
                actions[:, 10:20], actions_psm2, qpos_psm2.copy()
            )
            actions_psm2[:, 7] = np.clip(actions[:, 19], -0.698, 0.698)
        elif self.action_mode == "ego":
            actions_psm1 = processing.convert_actions_to_SE3_then_final_actions(
                actions[:, 0:3],
                processing.convert_6d_rot_to_quat(actions[:, 3:9]),
                qpos_psm1.copy(),
                actions[:, 9],
            )
            actions_psm2 = processing.convert_actions_to_SE3_then_final_actions(
                actions[:, 10:13],
                processing.convert_6d_rot_to_quat(actions[:, 13:19]),
                qpos_psm2.copy(),
                actions[:, 19],
            )
        else:
            raise NotImplementedError(f"Unsupported action mode: {self.action_mode}")

        return np.column_stack((actions_psm1, actions_psm2)).astype(np.float32)

    def postprocess_actions(
        self,
        policy_actions: torch.Tensor,
        current_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Convert predicted policy actions back into absolute robot commands.

        Args:
            policy_actions: Predicted action sequence in normalized policy
                space. May be batched or unbatched.
            current_pose: Current robot pose used as the reference frame for
                reconstructing absolute commands.

        Returns:
            Absolute robot actions in the dataset/runtime command format.
        """
        squeeze_batch = False
        if policy_actions.dim() == 2:
            policy_actions = policy_actions.unsqueeze(0)
            squeeze_batch = True
        if current_pose.dim() == 1:
            current_pose = current_pose.unsqueeze(0)

        denormalized_actions = self.denormalize_actions(policy_actions)
        denormalized_actions_np = denormalized_actions.detach().cpu().numpy()
        current_pose_np = current_pose.detach().cpu().numpy()

        absolute_actions_np = np.stack(
            [
                self._convert_single_policy_actions_to_absolute(pose, action)
                for pose, action in zip(current_pose_np, denormalized_actions_np)
            ],
            axis=0,
        )
        absolute_actions = torch.from_numpy(absolute_actions_np).to(
            device=policy_actions.device, dtype=policy_actions.dtype
        )
        if squeeze_batch:
            absolute_actions = absolute_actions.squeeze(0)
        return absolute_actions

    def forward(
        self,
        image,
        current_pose,
        actions=None,
        is_pad=None,
        command_text=None,
        return_policy_actions: bool = False,
    ):
        """Run the policy in training or inference mode.

        Training mode is selected by passing `actions`. In that case the method
        converts absolute actions into the policy representation, runs the ACT
        model, and returns a loss dictionary containing L1, KL, and total loss.

        In inference mode, the method samples from the ACT prior, predicts a
        sequence of policy actions, and either returns those raw policy-space
        predictions or converts them into absolute robot commands.

        Args:
            image: Image batch shaped `(B, num_cameras, C, H, W)` in `[0, 1]`.
            current_pose: Current robot pose used for action conversion.
            actions: Optional absolute action targets. Supplying this switches
                the method into training mode.
            is_pad: Optional padding mask aligned with `actions`.
            command_text: Optional command string or batch of strings for
                language-conditioned policies.
            return_policy_actions: In inference mode, return the raw policy
                action tensor instead of absolute robot actions.

        Returns:
            In training mode, a dictionary of loss tensors. In inference mode,
            either a tensor of predicted policy actions or a tensor of absolute
            robot actions.
        """
        env_state = None
        image = self.image_normalize(image)
        batch_size = image.shape[0]
        # since the dVRK is so inaccurate in an absolute setting, we set the absolute
        # qpos to zero so that this will not have an impact on the model
        model_qpos = torch.zeros(
            (batch_size, self.state_dim), dtype=image.dtype, device=image.device
        )
        command_embedding = self._encode_command_text(command_text, image.device)

        if actions is not None:  # training time
            processed_actions = self.prepare_actions_for_training(
                current_pose, actions, is_pad, image.device
            )
            processed_actions = processed_actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                model_qpos,
                image,
                env_state,
                processed_actions,
                is_pad,
                command_embedding=command_embedding,
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(processed_actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                model_qpos, image, env_state, command_embedding=command_embedding
            )  # no action, sample from prior
            if return_policy_actions:
                return a_hat
            return self.postprocess_actions(a_hat, current_pose)

    def configure_optimizers(self):
        """Return the optimizer constructed alongside the ACT model."""
        return self.optimizer

    def serialize(self):
        """Serialize policy state and lightweight metadata for checkpoints.

        Returns:
            A dictionary containing the full module state dict plus a small
            amount of metadata that is useful when restoring the policy later.
        """
        return {
            "state_dict": self.state_dict(),
            "policy_config": {
                "action_mode": self.action_mode,
                "norm_scheme": self.norm_scheme,
                "use_language": self.use_language,
                "language_encoder": self.language_encoder,
            },
            "stats_metadata": {
                "dataset_dir": self.stats_dataset_dir,
                "tissue_sample_ids_train": list(self.stats_tissue_sample_ids_train or []),
            },
        }

    def deserialize(self, model_dict):
        """Load policy weights from a serialized policy payload or state dict.

        Args:
            model_dict: Either the dictionary returned by `serialize()` or a
                raw PyTorch state dict.

        Returns:
            The result object returned by `load_state_dict`.
        """
        state_dict = model_dict
        if isinstance(model_dict, dict) and "state_dict" in model_dict:
            state_dict = model_dict["state_dict"]
            stats_metadata = model_dict.get("stats_metadata", {})
            self.stats_dataset_dir = stats_metadata.get("dataset_dir")
            self.stats_tissue_sample_ids_train = stats_metadata.get(
                "tissue_sample_ids_train"
            )
        load_result = self.load_state_dict(state_dict, strict=False)
        if "stats_mean" in state_dict:
            self.has_dataset_stats.fill_(True)
        return load_result

    def load_checkpoint(
        self,
        checkpoint_path,
        map_location: str | torch.device | None = None,
        load_optimizer: bool = False,
    ):
        """Load a training checkpoint from disk.

        This helper understands both the low-level training checkpoint format
        used in this repository and plain state-dict checkpoints.

        Args:
            checkpoint_path: Path to the checkpoint file.
            map_location: Optional `torch.load` map location.
            load_optimizer: Whether to also restore the optimizer state when it
                is present in the checkpoint.

        Returns:
            A tuple of `(checkpoint, load_result)` where `checkpoint` is the
            full deserialized checkpoint payload and `load_result` is the
            result from `load_state_dict`.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
            load_result = self.deserialize(checkpoint["policy_state"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            load_result = self.deserialize(checkpoint["model_state_dict"])
        else:
            load_result = self.deserialize(checkpoint)

        if (
            load_optimizer
            and isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint, load_result
