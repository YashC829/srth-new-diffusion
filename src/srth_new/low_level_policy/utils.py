from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from srth_new.general import constants
from srth_new.general.utils import dataset as general_dataset_utils
from srth_new.general.utils import processing
from srth_new.general.utils.processing import DatasetStats, compute_diffs
from srth_new.low_level_policy.dataset.low_level_dataset import (
    DvrkLerobotDataset,
)

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


import re
from pathlib import Path


def prune_checkpoints(checkpoint_dir, keep_every):
    """
    Prune checkpoints so that only one checkpoint is kept per keep_every bin.

    Assumes checkpoint filenames contain a training step integer, e.g.:
        ckpt_1000.pt
        checkpoint_step_5000.ckpt
        model-12000.pth

    For each bin [k*keep_every, (k+1)*keep_every), keeps the checkpoint
    with the largest step number and deletes the others.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_every: Bin size in training steps.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    if keep_every <= 0:
        raise ValueError("keep_every must be > 0")

    # Match the last integer before the file extension.
    step_pattern = re.compile(r"(\d+)(?=\.[^.]+$)")

    checkpoints = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue

        match = step_pattern.search(path.name)
        if match is None:
            continue

        step = int(match.group(1))
        checkpoints.append((step, path))

    if not checkpoints:
        return

    # Group checkpoints by bin index.
    bins = {}
    for step, path in checkpoints:
        bin_idx = step // keep_every
        bins.setdefault(bin_idx, []).append((step, path))

    # Keep the latest checkpoint in each bin, delete the rest.
    for _, ckpts in bins.items():
        ckpts.sort(key=lambda x: x[0])
        keep_step, keep_path = ckpts[-1]

        for step, path in ckpts[:-1]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def compute_dict_mean(
    epoch_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not epoch_dicts:
        raise ValueError("Cannot average an empty list of metrics.")

    result: dict[str, torch.Tensor] = {}
    for key in epoch_dicts[0]:
        value_sum = epoch_dicts[0][key]
        for epoch_dict in epoch_dicts[1:]:
            value_sum = value_sum + epoch_dict[key]
        result[key] = value_sum / len(epoch_dicts)
    return result


def restore_hydra_state_from_wandb(
    entity: str,
    project: str,
    run_id: str,
    root: str = ".hydra_restored",
) -> dict:
    """
    Restore the Hydra files from a previous W&B run.

    Returns:
        {
            "config": DictConfig,
            "hydra": DictConfig | None,
            "overrides": ListConfig | list[str] | None,
            "paths": {
                "config": Path,
                "hydra": Path | None,
                "overrides": Path | None,
            },
        }
    """
    run_path = f"{entity}/{project}/{run_id}"
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    def _restore(name: str):
        f = wandb.restore(
            name=name,
            run_path=run_path,
            root=str(root_path),
            replace=True,
        )
        return None if f is None else Path(f.name)

    config_path = _restore(".hydra/config.yaml")
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find '.hydra/config.yaml' in W&B run {run_path}"
        )

    hydra_path = _restore(".hydra/hydra.yaml")
    overrides_path = _restore(".hydra/overrides.yaml")

    restored = {
        "config": OmegaConf.load(config_path),
        "hydra": OmegaConf.load(hydra_path) if hydra_path else None,
        "overrides": OmegaConf.load(overrides_path) if overrides_path else None,
        "paths": {
            "config": config_path,
            "hydra": hydra_path,
            "overrides": overrides_path,
        },
    }
    return restored


def get_hydra_file_paths():
    hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"

    return [
        hydra_dir / "config.yaml",
        hydra_dir / "hydra.yaml",
        hydra_dir / "overrides.yaml",
        hydra_dir / "wandb_run_id.txt",
    ]


def save_wandb_run_id(run_id: str) -> Path:
    hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_id_path = hydra_dir / "wandb_run_id.txt"
    wandb_run_id_path.write_text(f"{run_id}\n", encoding="utf-8")
    return wandb_run_id_path


def wandb_setup(cfg: DictConfig) -> DictConfig:

    wandb_id = cfg.wandb.id
    resume_checkpoint = OmegaConf.select(cfg, "train.resume_checkpoint")

    # must save this here because if we are resuming, we will
    # load the config from the previously saved hydra config.
    # this will set wandb_resume to false, breaking downstream
    # logic
    wandb_resume = cfg.wandb.resume

    if wandb_resume:
        if not wandb_id:
            raise ValueError("cfg.wandb.id must be set when cfg.wandb.resume=True")

        if cfg.train.training_hydra_cfg_path:
            prior_training_cfg = OmegaConf.load(cfg.train.training_hydra_cfg_path)
        else:
            raise Exception(f"If wandb resume is set, train.training_hydra_cfg_path must also be set.")

        restored = restore_hydra_state_from_wandb(
            prior_training_cfg.wandb.entity,
            prior_training_cfg.wandb.project,
            wandb_id,
        )
        prior_training_cfg = restored["config"]
        resume_mode = "must"

        run = wandb.init(
            project=prior_training_cfg.wandb.project,
            entity=prior_training_cfg.wandb.entity,
            name=prior_training_cfg.wandb.name,
            id=wandb_id,
            resume=resume_mode,
            mode=prior_training_cfg.wandb.mode,
            config=OmegaConf.to_container(prior_training_cfg, resolve=True, throw_on_missing=True),
        )

    else:
        wandb_id = wandb_id or wandb.util.generate_id()
        resume_mode = None

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            id=wandb_id,
            resume=resume_mode,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
    cfg.wandb.resume = wandb_resume
    cfg.wandb.id = getattr(run, "id", wandb_id) or wandb_id
    save_wandb_run_id(getattr(run, "id", wandb_id) or wandb_id)

    # Only save Hydra files for fresh runs
    if not wandb_resume:
        for hydra_file in get_hydra_file_paths():
            if hydra_file.exists():
                run.save(
                    str(hydra_file),
                    base_path=str(hydra_file.parent.parent),
                    policy="now",
                )

    return cfg


def restore_hydra_cfg_from_wandb(entity: str, project: str, run_id: str) -> DictConfig:
    """
    Restore the resolved Hydra config from a previous W&B run.

    Assumes the config was saved under the same relative path:
    .hydra/config.yaml
    """
    run_path = f"{entity}/{project}/{run_id}"
    restore_root = Path(".wandb_restore")
    restore_root.mkdir(parents=True, exist_ok=True)

    restored = wandb.restore(
        ".hydra/config.yaml",
        run_path=run_path,
        root=str(restore_root),
        replace=True,
    )
    if restored is None:
        raise FileNotFoundError(
            f"Could not find '.hydra/config.yaml' in W&B run {run_path}"
        )

    restored_path = Path(restored.name)
    return OmegaConf.load(restored_path)


def load_dataset_stats(
    dataset_dir: str,
    tissue_sample_ids_train: List[int],
    phases: DictConfig,
    action_mode: str,
):

    general_dataset_utils.validate_selected_phases(phases)

    # we cache dataset stats for quick loading
    with open(constants.DATASET_STATS_CACHE_FILE, "r") as file:
        stats = json.load(file)

    # load cached stats, if possible
    for k, info in stats.items():
        if (
            info["dataset_dir"] == dataset_dir
            and info["tissue_sample_ids_train"] == tissue_sample_ids_train
            and info["action_mode"] == action_mode
        ):
            return DatasetStats(
                np.array(info["action_mean"]),
                np.array(info["action_std"]),
                np.array(info["action_min"]),
                np.array(info["action_max"]),
                info["dataset_dir"],
                info["tissue_sample_ids_train"],
                info["action_mode"],
            )

    # generate dataset stats
    log.info("Computing dataset statistics. This could take a few minutes...")

    temp_train_dataset = DvrkLerobotDataset(
        dataset_dir, tissue_sample_ids_train, phases
    )
    loader = DataLoader(temp_train_dataset, batch_size=12, shuffle=True)

    sum_ = None
    count = 0
    sample_count = 0

    # the dataset is random. it selects a random chunk of the trajectory from
    # each episode in the dataset. therefore, we will define a number of randomly
    # sampled trajectories and continue to sample from the data loader until the
    # chosen number of samples are aggregated for the statistics
    desired_samples = 10000
    pbar = tqdm(total=desired_samples, desc="Sampling actions")

    while sample_count < desired_samples:
        for data in loader:
            inputs = collect_data(data, torch.device("cpu"))
            current_pose = inputs["current_pose"]
            action_data = inputs["action"]
            is_pad = inputs["action_is_pad"]

            # the dataset stats will take place in the same action mode representation
            # as is used by the model. thus, we need to make the conversion here
            # because the dataset currently does not handle this
            processed_actions = processing.convert_action_batch_to_relative(
                current_pose, action_data, is_pad, action_mode
            )

            valid_mask = ~is_pad
            valid_actions = processed_actions[valid_mask]  # [K, 16]

            if valid_actions.numel() == 0:
                continue

            if sum_ is None:
                dim = valid_actions.shape[-1]
                device = valid_actions.device
                dtype = valid_actions.dtype

                sum_ = torch.zeros(dim, device=device, dtype=dtype)
                sum_sq = torch.zeros(dim, device=device, dtype=dtype)
                min_ = torch.full((dim,), float("inf"), device=device, dtype=dtype)
                max_ = torch.full((dim,), float("-inf"), device=device, dtype=dtype)

            sum_ += valid_actions.sum(dim=0)
            sum_sq += (valid_actions**2).sum(dim=0)
            count += valid_actions.shape[0]

            batch_min = valid_actions.min(dim=0).values
            batch_max = valid_actions.max(dim=0).values

            min_ = torch.minimum(min_, batch_min)
            max_ = torch.maximum(max_, batch_max)

            increment = action_data.shape[0]
            sample_count += increment
            pbar.update(increment)

            if sample_count >= desired_samples:
                break

    pbar.close()

    # Final statistics
    mean = sum_ / count
    var = sum_sq / count - mean**2
    var = torch.clamp(var, min=0.0)  # numerical stability
    std = torch.sqrt(var)

    # cache stats
    dataset_stats_cache_idx = len(stats.keys()) + 1
    stats[dataset_stats_cache_idx] = {
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "action_min": min_.tolist(),
        "action_max": max_.tolist(),
        "tissue_sample_ids_train": list(tissue_sample_ids_train),
        "dataset_dir": dataset_dir,
        "action_mode": action_mode,
    }
    with open(constants.DATASET_STATS_CACHE_FILE, "w") as file:
        json.dump(stats, file, indent=3)

    return DatasetStats(
        mean, std, min_, max_, dataset_dir, tissue_sample_ids_train, action_mode
    )


def load_dataloaders(cfg: DictConfig):
    log.info(f"Loading data from {cfg.repo_id}")
    dataset_stats = load_dataset_stats(
        cfg.repo_id, cfg.tissue_sample_ids_train, cfg.phases, cfg.action_mode
    )

    train_dataset = DvrkLerobotDataset(
        cfg.repo_id,
        cfg.tissue_sample_ids_train,
        cfg.phases,
        cfg.use_only_kp_annotated_data,
        cfg.history_chunk_size,
        cfg.future_chunk_size,
    )

    val_dataset = DvrkLerobotDataset(
        cfg.repo_id,
        cfg.tissue_sample_ids_val,
        cfg.phases,
        cfg.use_only_kp_annotated_data,
        cfg.history_chunk_size,
        cfg.future_chunk_size,
    )

    # we have removed the train sampler for now...
    # task_labels = [
    #     f"{Path(ep).parts[-3]}-{Path(ep).parts[-2]}"
    #     for ep in train_dataset.episode_dirs
    # ]
    # task_counts = Counter(task_labels)

    # # Compute weights based on task density in dataset distribution
    # weights = [1.0 / task_counts[task] for task in task_labels]
    # assert len(weights) == len(train_dataset)

    # train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset.episode_dirs), replacement=True)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
    #     pin_memory=True, num_workers=cfg.num_workers,  persistent_workers=True
    # )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, dataset_stats


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d


def collect_data(data, device: torch.device):
    (
        endoscope_img,
        lw_img,
        rw_img,
        current_pose,
        history_data,
        history_is_pad,
        action_data,
        action_is_pad,
        command_text,
        affordance_kp,
        tool_kp,
        has_affordance,
        original_ep_dir
    ) = data
    endoscope_img = endoscope_img.to(device)
    lw_img = lw_img.to(device)
    rw_img = rw_img.to(device)
    affordance_kp = affordance_kp.to(device)
    tool_kp = tool_kp.to(device)
    return {
        "endoscope_img": endoscope_img,
        "lw_img": lw_img,
        "rw_img": rw_img,
        "current_pose": current_pose,
        "action_history": history_data,
        "action_history_is_pad": history_is_pad,
        "action": action_data,
        "action_is_pad": action_is_pad.to(device),
        "command_text": list(command_text),
        "affordance_kp": affordance_kp,
        "tool_kp": tool_kp,
        "has_affordance": has_affordance,
        "original_ep_dir": original_ep_dir
    }
