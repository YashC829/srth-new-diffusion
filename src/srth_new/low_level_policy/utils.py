from __future__ import annotations

import json
from collections import Counter
import math
from pathlib import Path
import random

from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb

from srth_new.general import constants
from srth_new.general.utils import DatasetStats
from srth_new.low_level_policy.dataset.normalization import compute_diffs
from srth_new.low_level_policy.dataset.low_level_dataset import EpisodicDatasetDvrkGeneric
from srth_new.general import constants

import logging
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


def compute_dict_mean(epoch_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)

    return [
        hydra_run_dir / ".hydra/config.yaml",
        hydra_run_dir / ".hydra/hydra.yaml",
        hydra_run_dir / ".hydra/overrides.yaml",
    ]

def wandb_setup(cfg: DictConfig) -> DictConfig:
    wandb_id = cfg.wandb.id

    # must save this here because if we are resuming, we will
    # load the config from the previously saved hydra config.
    # this will set wandb_resume to false, breaking downstream
    # logic
    wandb_resume = cfg.wandb.resume

    if wandb_resume:
        if not wandb_id:
            raise ValueError("cfg.wandb.id must be set when cfg.wandb.resume=True")

        restored = restore_hydra_state_from_wandb(
            cfg.wandb.entity,
            cfg.wandb.project,
            wandb_id,
        )
        cfg = restored["config"]
        resume_mode = "must"
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

def load_dataset_stats(tissue_sample_ids_train, dataset_dir):

    # we cache dataset stats for quick loading
    with open(constants.DATASET_STATS_CACHE_FILE, "r") as file:
        stats = json.load(file)

    # load cached stats, if possible
    for k, info in stats.items():
        if (info["dataset_dir"] == dataset_dir 
            and info["tissue_sample_ids_train"] == tissue_sample_ids_train):
            return DatasetStats(
                np.array(info["mean"]),
                np.array(info["std"]),
                np.array(info["min"]),
                np.array(info["max"]),
                info["dataset_dir"],
                info["tissue_sample_ids_train"]
            )
    
    # generate dataset stats
    log.info('Computing dataset statistics. This could take a few minutes...')
    mean, std, min, max = compute_diffs(tissue_sample_ids_train, dataset_dir)

    # cache stats
    dataset_stats_cache_idx = len(stats.keys()) + 1
    stats[dataset_stats_cache_idx] = {
        "mean": mean.tolist(), 
        "std": std.tolist(), 
        "min": min.tolist(), 
        "max": max.tolist(), 
        "tissue_sample_ids_train": list(tissue_sample_ids_train), 
        "dataset_dir": dataset_dir
    }
    with open(constants.DATASET_STATS_CACHE_FILE, "w") as file:
        json.dump(stats, file, indent=3)

    return DatasetStats(mean, std, min, max, dataset_dir, tissue_sample_ids_train)


def load_dataloaders(cfg: DictConfig):
    log.info(f"Loading data from {cfg.dataset_dir}")
    
    # obtain train test split
    train_indices = np.random.permutation(cfg.num_episodes_train)
    val_indices = np.random.permutation(cfg.num_episodes_val)

    # TODO: All are hardcoded... This is taken from the original code. There are
    # other hardcoded things that make changing this a bit tedious and nontrivial.
    # For now, we will just hardcode this here as it works, but if we change any
    # of the names anywhere or change the data collection naming convention, this
    # will break. I will define these for now in the general.constants.py file, 
    # but an overhaul of all naming would be nice for organization purposes.
    camera_names = constants.LOW_LEVEL_DATASET_CAMERA_NAMES
    camera_file_suffixes = constants.LOW_LEVEL_DATASET_CAMERA_SUFFIXES

    dataset_stats = load_dataset_stats(cfg.tissue_sample_ids_train, cfg.dataset_dir)

    temp_debug_dataset_stats = load_dataset_stats([1, 2], cfg.dataset_dir)

    train_dataset = EpisodicDatasetDvrkGeneric(
        train_indices,
        cfg.tissue_sample_ids_train,
        cfg.dataset_dir,
        temp_debug_dataset_stats,
        camera_names,
        camera_file_suffixes,
        cfg.action_mode,
        cfg.norm_scheme,
        cfg.chunk_size,
        cfg.use_language,
        cfg.language_encoder,
        cfg.use_auto_label
    )

    val_dataset = EpisodicDatasetDvrkGeneric(
        val_indices,
        cfg.tissue_sample_ids_val,
        cfg.dataset_dir,
        dataset_stats,
        camera_names,
        camera_file_suffixes,
        cfg.action_mode,
        cfg.norm_scheme,
        cfg.chunk_size,
        cfg.use_language,
        cfg.language_encoder,
        cfg.use_auto_label
    )

    task_labels = train_dataset.sample_task_labels
    task_counts = Counter(task_labels)

    # Compute weights based on task density in dataset distribution
    weights = [1.0 / task_counts[task] for task in task_labels]

    train_sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=cfg.num_workers,  persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, pin_memory=True, 
        num_workers=cfg.num_workers,  persistent_workers=True
    )

    return train_dataloader, val_dataloader

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
    image_data, qpos_data, action_data, is_pad, command_embedding = data
    return (
        image_data.to(device), 
        qpos_data.to(device), 
        action_data.to(device), 
        is_pad.to(device), 
        command_embedding.to(device)
    )

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result