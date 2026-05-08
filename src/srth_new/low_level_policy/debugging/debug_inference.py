from __future__ import annotations

import logging
import os
import pdb
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from srth_new.low_level_policy import utils

log = logging.getLogger(__name__)


def resume_training_state(
    train_cfg: DictConfig,
    policy,
    scheduler: LambdaLR,
    device: torch.device,  # type: ignore
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


def log_to_wandb(metrics, summary_prefix, epoch, step):
    # Compute Mean Metrics and Log to WandB and Local Files
    avg_metrics = utils.compute_dict_mean(metrics)
    step_sumary = {f"{summary_prefix}/{k}": v.item() for k, v in avg_metrics.items()}
    wandb.log(step_sumary, step=step)
    log.info(f"{summary_prefix} - Epoch/Step: {epoch}/{step} - Summary: {step_sumary}")


def validate(cfg: DictConfig):
    if cfg.wandb.resume and not cfg.train.resume_checkpoint:
        raise Exception(
            "wandb.resume=true but train.resume_checkpoint is unset. incompatible behavior"
        )


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

    epoch = starting_step // len(train_loader)

    pbar = tqdm(
        total=train_cfg.num_train_steps,
        initial=training_step,
        desc="Training",
        unit="step",
    )

    while training_step < train_cfg.num_train_steps:

        for data in val_loader:
            inputs = utils.collect_data(data, device)
            # processed_actions = policy.prepare_actions_for_training(
            #     inputs["current_pose"].to("cuda"),
            #     inputs["action"].to("cuda"),
            #     inputs["action_is_pad"].to("cuda"),
            # )

            policy.eval()

            loss_dict = policy(
                inputs["endoscope_img"],
                inputs["lw_img"],
                inputs["rw_img"],
                inputs["current_pose"],
                inputs["action"],
                inputs["action_is_pad"],
                command_text=inputs["command_text"],
                return_policy_actions=False,
            )

            print(loss_dict)

            policy.train()

            loss_dict = policy(
                inputs["endoscope_img"],
                inputs["lw_img"],
                inputs["rw_img"],
                inputs["current_pose"],
                inputs["action"],
                inputs["action_is_pad"],
                command_text=inputs["command_text"],
                return_policy_actions=False,
            )

            print(loss_dict)
            print("\n\n")

            # absolute_action, processed_action = policy.forward_debug(
            #     inputs["endoscope_img"],
            #     inputs["lw_img"],
            #     inputs["rw_img"],
            #     inputs["current_pose"],
            #     command_text=inputs["command_text"],
            #     return_policy_actions=False,
            # )
            # log.info(f"Processed action: {processed_action[:, 0, -1]}")
            # log.info(f"GT Processed action: {processed_actions[:, 0, -1]}")
            # log.info(f"Absolute action: {absolute_action[:, 0, -1]}")
            # log.info(f"GT Absolute action: {inputs['action'][:, 0, -1]}")

            # processed_diff = processed_action[:, 0, -1].to(
            #     torch.device("cuda")
            # ) - processed_actions[:, 0, -1].to(torch.device("cuda"))
            # log.info(f"Processed diff jaw angle: {processed_diff}")
            #
            # processed_diff = processed_action[:, 0, 8].to(
            #     torch.device("cuda")
            # ) - processed_actions[:, 0, 8].to(torch.device("cuda"))
            # log.info(f"Processed diff x: {processed_diff}")
            #
            # processed_diff = processed_action[:, 0, 9].to(
            #     torch.device("cuda")
            # ) - processed_actions[:, 0, 9].to(torch.device("cuda"))
            # log.info(f"Processed diff y: {processed_diff}")
            #
            # absolute_diff = absolute_action[:, 0, -1].to(torch.device("cuda")) - inputs[
            #     "action"
            # ][:, 0, -1].to(torch.device("cuda"))
            # log.info(f"Absolute diff jaw angle: {absolute_diff}")
            #
            # absolute_diff = absolute_action[:, 0, 8].to(torch.device("cuda")) - inputs[
            #     "action"
            # ][:, 0, 8].to(torch.device("cuda"))
            # log.info(f"Absolute diff x: {absolute_diff}")
            #
            # absolute_diff = absolute_action[:, 0, 9].to(torch.device("cuda")) - inputs[
            #     "action"
            # ][:, 0, 9].to(torch.device("cuda"))
            # log.info(f"Absolute diff y: {absolute_diff}")

            # pdb.set_trace()

            # run validation and log loss metrics to wandb
            if training_step % train_cfg.validate_every == 0:
                with torch.inference_mode():
                    policy.eval()
                    val_batches = 0
                    val_sample_size = min(1000, len(val_loader))
                    for data in val_loader:
                        inputs = utils.collect_data(data, device)
                        forward_dict = policy(**inputs)

                        val_metrics.append(utils.detach_dict(forward_dict))
                        val_batches += 1

                        if val_batches >= val_sample_size:
                            break

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
    version_base=None,
    config_path="../../../../conf/low_level_policy",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    utils.set_seed(int(cfg.seed))
    device = utils.resolve_device(str(cfg.device))
    train_loader, val_loader, dataset_stats = utils.load_dataloaders(cfg.dataloader)
    policy = instantiate(cfg.policy).to(device)
    policy.set_dataset_stats(dataset_stats)
    optimizer = policy.configure_optimizers()
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.num_warmup_steps,
        num_training_steps=cfg.train.num_train_steps,
    )
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
