from __future__ import annotations

import os
import sys
from copy import deepcopy
import logging
from pathlib import Path
from typing import Tuple

import hydra
import torch
import wandb
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from srth_new.low_level_policy.models.dvrk_policy import DVRKPolicy
from srth_new.low_level_policy import utils

log = logging.getLogger(__name__)


def resume_training_state(
    train_cfg: DictConfig,
    policy: DVRKPolicy,
    scheduler: LambdaLR,
    device: torch.device,  # type: ignore
) -> int:
    resume_checkpoint = train_cfg.resume_checkpoint

    checkpoint_path = Path(to_absolute_path(str(resume_checkpoint)))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint, _ = policy.load_checkpoint(
        checkpoint_path,
        map_location=device,
        load_optimizer=True,
    )

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_step = int(checkpoint.get("step", -1)) + 1
    log.info("Resumed training state from %s at step %s", checkpoint_path, start_step)

    if start_step >= train_cfg.num_train_steps:
        log.info(
            "Resume checkpoint is already at or beyond num_train_steps (%s >= %s); exiting.",
            start_step,
            train_cfg.num_train_steps,
        )
        exit()

    return start_step


def log_to_wandb(metrics, summary_prefix, epoch, step):
    # Compute Mean Metrics and Log to WandB and Local Files
    avg_metrics = utils.compute_dict_mean(metrics)
    step_sumary = {f"{summary_prefix}/{k}": v.item() for k, v in avg_metrics.items()}
    wandb.log(step_sumary, step=step)
    log.info(f"{summary_prefix} - Epoch/Step: {epoch}/{step} - Summary: {step_sumary}")


def validate(cfg: DictConfig):
    if (
        not cfg.wandb.resume 
        and not cfg.train.resume_checkpoint 
        and not cfg.train.training_hydra_cfg_path
    ):
        return
    
    elif (
        cfg.wandb.resume
        and cfg.train.resume_checkpoint
        and cfg.train.training_hydra_cfg_path
    ):
        return

    else:
        raise Exception(f"Invalid hydra training config. It seems you might be "
                        "trying to resume training from a prior config. "
                        "To train from scratch, cfg.wandb.resume, cfg.train.resume_checkpoint,"
                        "and cfg.train.training_hydra_cfg_path must all be either null or false. "
                        "To resume, these parameters must all be set. Right now, these"
                        "parameters are partially set, therefore your intention is ambiguous and,"
                        "thus, invalid.")


def run_training(
    train_cfg,
    policy,
    train_loader,
    val_loader,
    device,
    optimizer,
    scheduler,
    starting_step,
):
    train_metrics = list()
    val_metrics = list()
    training_step = starting_step
    policy.train()

    epoch = starting_step // len(train_loader)

    pbar = tqdm(
        total=train_cfg.num_train_steps,
        initial=training_step,
        desc="Training",
        unit="step",
    )

    while training_step < train_cfg.num_train_steps:

        # run training
        for data in train_loader:
            inputs = utils.collect_data(data, device)
            forward_dict = policy(**inputs)

            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_metrics.append(utils.detach_dict(forward_dict))
            training_step += 1
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # run validation and log loss metrics to wandb
            if training_step % train_cfg.validate_every == 0:
                with torch.inference_mode():
                    policy.eval()
                    val_batches = 0
                    val_sample_size = min(100, len(val_loader))
                    for data in val_loader:
                        inputs = utils.collect_data(data, device)
                        forward_dict = policy(**inputs)

                        val_metrics.append(utils.detach_dict(forward_dict))
                        val_batches += 1

                        if val_batches >= val_sample_size:
                            break
                policy.train()

                # log to wandb and clear out metrics
                log_to_wandb(train_metrics, "train", epoch, training_step)
                log_to_wandb(val_metrics, "val", epoch, training_step)
                train_metrics = list()
                val_metrics = list()

            # Prune Checkpoints
            utils.prune_checkpoints(train_cfg.checkpoint_dir, train_cfg.keep_every)

            # Save Checkpoints
            if training_step % train_cfg.save_every == 0:
                os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
                ckpt_path = Path(train_cfg.checkpoint_dir).joinpath(
                    f"train_step_{training_step}.ckpt"
                )
                torch.save(
                    {
                        "policy_state": policy.serialize(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": training_step,
                    },
                    ckpt_path,
                )

            if training_step >= train_cfg.num_train_steps:
                break

        epoch += 1

    pbar.close()


@hydra.main(
    version_base=None, config_path="../../../conf/low_level_policy", config_name="train"
)
def main(cfg: DictConfig) -> None:
    validate(cfg)
    cfg = utils.wandb_setup(cfg)
    utils.set_seed(int(cfg.seed))
    device = utils.resolve_device(str(cfg.device))
    train_loader, val_loader, dataset_stats = utils.load_dataloaders(cfg.dataloader)
    policy_cfg = cfg.policy
    if cfg.train.training_hydra_cfg_path:
        policy_cfg = OmegaConf.load(cfg.train.training_hydra_cfg_path).policy
    policy = instantiate(policy_cfg).to(device)
    policy.set_dataset_stats(dataset_stats)
    optimizer = policy.configure_optimizers()
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.num_warmup_steps,
        num_training_steps=cfg.train.num_train_steps,
    )
    start_step = 0
    if cfg.train.resume_checkpoint:
        start_step = resume_training_state(cfg.train, policy, scheduler, device)
    run_training(
        cfg.train,
        policy,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        start_step,
    )


if __name__ == "__main__":
    main()