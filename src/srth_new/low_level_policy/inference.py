from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .data import build_dataloader, discover_episode_paths, load_stats, split_episode_paths
from .policy import ACTPolicy
from .utils import resolve_device, set_seed


def build_policy_config(cfg: DictConfig) -> dict[str, object]:
    return {
        "backbone": cfg.policy.backbone,
        "pretrained_backbone": cfg.policy.pretrained_backbone,
        "lr": cfg.policy.lr,
        "lr_backbone": cfg.policy.lr_backbone,
        "weight_decay": cfg.policy.weight_decay,
        "kl_weight": cfg.policy.kl_weight,
        "hidden_dim": cfg.policy.hidden_dim,
        "dim_feedforward": cfg.policy.dim_feedforward,
        "enc_layers": cfg.policy.enc_layers,
        "dec_layers": cfg.policy.dec_layers,
        "nheads": cfg.policy.nheads,
        "dropout": cfg.policy.dropout,
        "position_embedding": cfg.policy.position_embedding,
        "camera_names": list(cfg.task.camera_names),
        "state_dim": cfg.task.state_dim,
        "action_dim": cfg.task.action_dim,
        "num_queries": cfg.task.chunk_size,
    }


def denormalize_actions(actions: torch.Tensor, stats: dict[str, object]) -> torch.Tensor:
    action_mean = torch.from_numpy(stats["action_mean"]).view(1, 1, -1)
    action_std = torch.from_numpy(stats["action_std"]).view(1, 1, -1)
    return actions.cpu() * action_std + action_mean


@hydra.main(version_base=None, config_path="../../conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    dataset_dir = Path(to_absolute_path(cfg.task.dataset_dir))
    checkpoint_path = Path(to_absolute_path(cfg.checkpoint_path))
    stats_path = (
        Path(to_absolute_path(cfg.stats_path))
        if cfg.stats_path is not None
        else checkpoint_path.with_name("dataset_stats.pkl")
    )

    episode_paths = discover_episode_paths(dataset_dir, cfg.task.max_episodes)
    train_paths, val_paths = split_episode_paths(episode_paths, float(cfg.task.val_ratio), int(cfg.seed))
    selected_paths = train_paths if str(cfg.split) == "train" else val_paths
    stats = load_stats(stats_path)

    dataloader = build_dataloader(
        episode_paths=selected_paths,
        camera_names=list(cfg.task.camera_names),
        stats=stats,
        chunk_size=int(cfg.task.chunk_size),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        train=False,
    )

    policy = ACTPolicy(build_policy_config(cfg)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.deserialize(checkpoint["model_state_dict"])
    policy.eval()

    outputs = []
    with torch.inference_mode():
        for batch_idx, (images, qpos, actions, is_pad) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            qpos = qpos.to(device, non_blocking=True)

            predicted_actions = policy(qpos=qpos, image=images)
            predicted_actions = denormalize_actions(predicted_actions, stats)
            target_actions = denormalize_actions(actions, stats)

            result = {
                "predicted_actions": predicted_actions,
                "target_actions": target_actions,
                "is_pad": is_pad,
            }
            outputs.append(result)
            print(
                f"Batch {batch_idx}: predicted_actions={tuple(predicted_actions.shape)} "
                f"target_actions={tuple(target_actions.shape)}"
            )

            if batch_idx + 1 >= int(cfg.num_batches):
                break

    if cfg.output_path is not None:
        output_path = Path(to_absolute_path(cfg.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, output_path)
        print(f"Saved inference outputs to {output_path}")


if __name__ == "__main__":
    main()
