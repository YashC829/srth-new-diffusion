from __future__ import annotations

from pathlib import Path

import os
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
import wandb

from srth_new.low_level_policy import utils

import logging
log = logging.getLogger(__name__)


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
                image_data, qpos_data, action_data, is_pad, command_embedding = utils.collect_data(data, device)
                forward_dict = policy(qpos_data, image_data, action_data, is_pad, command_embedding)

                metrics.append(utils.detach_dict(forward_dict))
    
    # Training Step
    else:
        policy.train()
        for data in dataloader:
            image_data, qpos_data, action_data, is_pad, command_embedding = utils.collect_data(data, device)
            forward_dict = policy(qpos_data, image_data, action_data, is_pad, command_embedding)

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
                "model_state_dict": policy.serialize(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
            },
            ckpt_path
        )

class FrozenDataLoader:
    def __init__(self, path, map_location="cpu"):
        self.batches = torch.load(path, map_location=map_location)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

@hydra.main(version_base=None, config_path="../../../conf/low_level_policy", config_name="config")
def main(cfg: DictConfig) -> None:
    utils.set_seed(1)
    cfg = utils.wandb_setup(cfg)
    train_loader, val_loader = utils.load_dataloaders(cfg.dataloader)
    policy = instantiate(cfg.policy)
    optimizer = policy.configure_optimizers()
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg.train.num_warmup_steps, 
        num_training_steps=cfg.train.num_train_steps
    )

    for step in range(cfg.train.num_train_steps):
        run_policy_step(
            cfg.train, policy, train_loader, 
            torch.device(cfg.device), step, optimizer, scheduler
        )
        if step % cfg.train.validate_every == 0:
            run_policy_step(
                cfg.train, policy, val_loader,
                torch.device(cfg.device), step, optimizer, scheduler, run_val=True
            )


if __name__ == "__main__":
    main()
