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
        self.stats_mean.copy_(torch.as_tensor(dataset_stats.action_mean, dtype=torch.float32))
        self.stats_std.copy_(torch.as_tensor(dataset_stats.action_std, dtype=torch.float32))
        self.stats_min.copy_(torch.as_tensor(dataset_stats.action_min, dtype=torch.float32))
        self.stats_max.copy_(torch.as_tensor(dataset_stats.action_max, dtype=torch.float32))
        self.has_dataset_stats.fill_(True)
        self.stats_dataset_dir = dataset_stats.dataset_dir
        self.stats_tissue_sample_ids_train = list(dataset_stats.tissue_sample_ids_train)
        if dataset_stats.action_mode != self.action_mode:
            raise Exception(
                f"Action mode used to collect dataset stats ({dataset_stats.action_mode}) "
                "is not the same as action mode required by the model ({self.action_mode})"
            )

    def export_dataset_stats(self) -> processing.DatasetStats:
        """Return the currently attached dataset statistics as a dataclass."""
        if not bool(self.has_dataset_stats.item()):
            raise RuntimeError("Dataset statistics have not been set on the policy.")

        return processing.DatasetStats(
            action_mean=self.stats_mean.detach().cpu().numpy().copy(),
            action_std=self.stats_std.detach().cpu().numpy().copy(),
            action_min=self.stats_min.detach().cpu().numpy().copy(),
            action_max=self.stats_max.detach().cpu().numpy().copy(),
            dataset_dir=self.stats_dataset_dir,
            tissue_sample_ids_train=list(self.stats_tissue_sample_ids_train or []),
            action_mode=self.action_mode
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

    def prepare_actions_for_training(
        self,
        current_pose: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor
    ) -> torch.Tensor:
        """Take current absolute actions and pose and do the following:
            1. Convert absolute actions to relative actions according to the chosen
                self.action_mode
            2. Normalize the relative actions according to chosen self.norm_scheme
        """
        policy_actions = processing.convert_action_batch_to_relative(
            current_pose, actions, is_pad, self.action_mode
        )
        policy_actions = policy_actions.to(actions.device)
        policy_actions = self.normalize_actions(policy_actions)
        policy_actions = policy_actions.masked_fill(is_pad.unsqueeze(-1), 0.0)
        return policy_actions

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
                processing.convert_single_policy_actions_to_absolute(pose, action, self.action_mode)
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
