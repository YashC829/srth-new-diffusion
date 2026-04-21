from __future__ import annotations

from pathlib import Path

import os
import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
import wandb

from srth_new.low_level_policy import utils

import logging
log = logging.getLogger(__name__)


def resume_training_state(
    train_cfg: DictConfig,
    policy,
    scheduler: LambdaLR,
    device: torch.device,
) -> int:
    resume_checkpoint = train_cfg.resume_checkpoint
    if not resume_checkpoint:
        return 0

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


def run_policy_step(
    train_cfg: DictConfig,
    policy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    run_val: bool = False
):
    
    metrics = list()

    # Validation Step
    if run_val:
        with torch.inference_mode():
            policy.eval()
            for data in dataloader:
                image_data, current_pose_data, action_data, is_pad, command_text = utils.collect_data(data, device)
                forward_dict = policy(image_data, current_pose_data, action_data, is_pad, command_text)

                metrics.append(utils.detach_dict(forward_dict))
    
    # Training Step
    else:
        policy.train()
        for data in dataloader:
            img_stack, current_pose_data, action_data, is_pad, command_text, endoscope_img_spatial_tf_only = utils.collect_data(data, device)
            forward_dict = policy(img_stack, current_pose_data, action_data, is_pad, command_text, endoscope_img_spatial_tf_only)

            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            metrics.append(utils.detach_dict(forward_dict))
        scheduler.step() # scheduler done once per training step (full pass through dataloader)

    # Compute Mean Metrics and Log to WandB and Local Files
    avg_metrics = utils.compute_dict_mean(metrics)
    summary_prefix = "train" if not run_val else "val"
    epoch_summary = {f"{summary_prefix}/{k}": v.item() for k, v in avg_metrics.items()}
    wandb.log(epoch_summary, step=step)
    log.info(f"{summary_prefix} - Step: {step} - Summary: {epoch_summary}")

    # Prune Checkpoints
    utils.prune_checkpoints(train_cfg.checkpoint_dir, train_cfg.keep_every)

    # Save Checkpoints
    if step % train_cfg.save_every == 0:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = Path(train_cfg.checkpoint_dir).joinpath(f"train_step_{step}.ckpt")
        torch.save(
            {
                "policy_state": policy.serialize(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
            },
            ckpt_path
        )


def validate(cfg: DictConfig):
    if cfg.wandb.resume and not cfg.train.resume_checkpoint:
        raise Exception(
            "wandb.resume=true but train.resume_checkpoint is unset. incompatible behavior"
        )


@hydra.main(version_base=None, config_path="../../../conf/low_level_policy", config_name="train")
def main(cfg: DictConfig) -> None:
    validate(cfg)
    cfg = utils.wandb_setup(cfg)
    utils.set_seed(int(cfg.seed))
    device = utils.resolve_device(str(cfg.device))
    train_loader, val_loader, dataset_stats = utils.load_dataloaders(cfg.dataloader)
    policy = instantiate(cfg.policy).to(device)
    policy.set_dataset_stats(dataset_stats)
    optimizer = policy.configure_optimizers()
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.num_warmup_steps, 
        num_training_steps=cfg.train.num_train_steps
    )
    start_step = resume_training_state(cfg.train, policy, scheduler, device)

    for step in range(start_step, cfg.train.num_train_steps):
        run_policy_step(
            cfg.train, policy, train_loader, 
            device, step, optimizer, scheduler
        )
        if step % cfg.train.validate_every == 0:
            run_policy_step(
                cfg.train, policy, val_loader,
                device, step, optimizer, scheduler, run_val=True
            )


if __name__ == "__main__":
    main()
