from __future__ import annotations

import numpy as np
import torch
from torch import nn

from srth_new.general.utils import processing


class DVRKPolicy(nn.Module):
    """Shared dVRK policy utilities for action normalization and conversion."""

    _ROTATION_COLUMN_SLICES: tuple[tuple[int, int], ...] = ((3, 9), (13, 19))

    def __init__(
        self,
        action_dim: int,
        action_mode: str = "hybrid_relative",
        norm_scheme: str = "std",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_mode = str(action_mode)
        self.norm_scheme = str(norm_scheme)

        self.register_buffer("stats_mean", torch.zeros(self.action_dim))
        self.register_buffer("stats_std", torch.ones(self.action_dim))
        self.register_buffer("stats_min", torch.zeros(self.action_dim))
        self.register_buffer("stats_max", torch.zeros(self.action_dim))
        self.register_buffer("has_dataset_stats", torch.tensor(False))
        self.stats_dataset_dir: str | None = None
        self.stats_tissue_sample_ids_train: list[int] | None = None

    def set_dataset_stats(self, dataset_stats: processing.DatasetStats) -> None:
        """Attach dataset statistics used for action normalization."""
        self.stats_mean.copy_(torch.as_tensor(dataset_stats.mean, dtype=torch.float32))
        self.stats_std.copy_(torch.as_tensor(dataset_stats.std, dtype=torch.float32))
        self.stats_min.copy_(torch.as_tensor(dataset_stats.min, dtype=torch.float32))
        self.stats_max.copy_(torch.as_tensor(dataset_stats.max, dtype=torch.float32))
        self.has_dataset_stats.fill_(True)
        self.stats_dataset_dir = dataset_stats.dataset_dir
        self.stats_tissue_sample_ids_train = list(dataset_stats.tissue_sample_ids_train)

    def export_dataset_stats(self) -> processing.DatasetStats:
        """Return the currently attached dataset statistics as a dataclass."""
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

    def _require_dataset_stats(self) -> None:
        if not bool(self.has_dataset_stats.item()):
            raise RuntimeError(
                "Dataset statistics must be set on the policy before normalization."
            )

    @classmethod
    def _preserve_rotation_columns(
        cls,
        normalized: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        for start, end in cls._ROTATION_COLUMN_SLICES:
            if raw.shape[-1] >= end:
                normalized[..., start:end] = raw[..., start:end]
        return normalized

    def _stats_for(self, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.stats_mean.to(device=actions.device, dtype=actions.dtype),
            self.stats_std.to(device=actions.device, dtype=actions.dtype),
            self.stats_min.to(device=actions.device, dtype=actions.dtype),
            self.stats_max.to(device=actions.device, dtype=actions.dtype),
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize policy-space actions with the configured statistics scheme."""
        self._require_dataset_stats()
        stats_mean, stats_std, stats_min, stats_max = self._stats_for(actions)

        if self.norm_scheme == "std":
            normalized = (actions - stats_mean) / stats_std
        elif self.norm_scheme == "min_max":
            normalized = (actions - stats_min) / (stats_max - stats_min) * 2 - 1
        else:
            raise NotImplementedError(f"Unsupported norm scheme: {self.norm_scheme}")

        return self._preserve_rotation_columns(normalized, actions)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert `normalize_actions` using the stored dataset statistics."""
        self._require_dataset_stats()
        stats_mean, stats_std, stats_min, stats_max = self._stats_for(actions)

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
            diff_psm1 = processing.compute_diff_actions_relative_endoscope(
                qpos_psm1,
                actions_psm1,
            )
            diff_psm2 = processing.compute_diff_actions_relative_endoscope(
                qpos_psm2,
                actions_psm2,
            )
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
        """Convert absolute dataset actions into normalized policy targets."""
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
                pose,
                valid_actions,
            )
            policy_actions_np[batch_idx, valid_mask] = converted_actions

        policy_actions = torch.from_numpy(policy_actions_np).to(
            device=device,
            dtype=torch.float32,
        )
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
                actions[:, 0:10],
                actions_psm1,
                qpos_psm1.copy(),
            )
            actions_psm1[:, 7] = np.clip(actions[:, 9], -0.698, 0.698)

            actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm2[:, 0:3] = qpos_psm2[0:3] + actions[:, 10:13]
            actions_psm2 = processing.convert_delta_6d_to_taskspace_quat(
                actions[:, 10:20],
                actions_psm2,
                qpos_psm2.copy(),
            )
            actions_psm2[:, 7] = np.clip(actions[:, 19], -0.698, 0.698)
        elif self.action_mode == "relative_endoscope":
            actions_psm1 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm1[:, 0:3] = qpos_psm1[0:3] + actions[:, 0:3]
            actions_psm1 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(
                actions[:, 0:10],
                actions_psm1,
                qpos_psm1.copy(),
            )
            actions_psm1[:, 7] = np.clip(actions[:, 9], -0.698, 0.698)

            actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
            actions_psm2[:, 0:3] = qpos_psm2[0:3] + actions[:, 10:13]
            actions_psm2 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(
                actions[:, 10:20],
                actions_psm2,
                qpos_psm2.copy(),
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
        """Convert predicted policy actions back into absolute robot commands."""
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
            device=policy_actions.device,
            dtype=policy_actions.dtype,
        )
        if squeeze_batch:
            absolute_actions = absolute_actions.squeeze(0)
        return absolute_actions

    def _serialize_policy_config(self) -> dict[str, object]:
        """Return checkpointed policy configuration for this policy type."""
        return {
            "action_mode": self.action_mode,
            "norm_scheme": self.norm_scheme,
        }

    def _serialize_checkpoint_metadata(self) -> dict[str, object]:
        """Return subclass-specific checkpoint metadata."""
        return {}

    def _restore_checkpoint_metadata(self, model_dict: dict[str, object]) -> None:
        """Restore subclass-specific checkpoint metadata."""
        pass

    def serialize(self) -> dict[str, object]:
        """Serialize policy state and lightweight metadata for checkpoints."""
        payload: dict[str, object] = {
            "state_dict": self.state_dict(),
            "policy_config": self._serialize_policy_config(),
            "stats_metadata": {
                "dataset_dir": self.stats_dataset_dir,
                "tissue_sample_ids_train": list(self.stats_tissue_sample_ids_train or []),
            },
        }
        payload.update(self._serialize_checkpoint_metadata())
        return payload

    def deserialize(self, model_dict):
        """Load policy weights from a serialized policy payload or state dict."""
        state_dict = model_dict
        if isinstance(model_dict, dict) and "state_dict" in model_dict:
            state_dict = model_dict["state_dict"]
            stats_metadata = model_dict.get("stats_metadata") or {}
            self.stats_dataset_dir = stats_metadata.get("dataset_dir")
            self.stats_tissue_sample_ids_train = stats_metadata.get(
                "tissue_sample_ids_train"
            )
            self._restore_checkpoint_metadata(model_dict)
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
        """Load a training checkpoint from disk."""
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
            if not hasattr(self, "optimizer"):
                raise RuntimeError(
                    "Cannot load optimizer state because this policy does not define one."
                )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint, load_result
