from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from srth_new.general.third_party.EndoSynth.endosynth.utils import depth2rgb
from srth_new.low_level_policy import utils

import logging
log = logging.getLogger(__name__)

from srth_new.low_level_policy.dataset.low_level_dataset_new import EpisodicDatasetDvrkGeneric


def _unnormalize_rgb_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def _depth_to_rgb(depth: torch.Tensor) -> np.ndarray:
    depth = depth.squeeze().detach().cpu().clamp(0.0, 1.0)
    return depth2rgb(depth.numpy(), 0.0, 1.0).astype(np.uint8)


@hydra.main(version_base=None, config_path="./conf/low_level_policy", config_name="train")
def main(cfg: DictConfig) -> None:
    from srth_new.general import constants
    num_episodes_train = 100
    num_episodes_val = 28
    train_indices = np.random.permutation(num_episodes_train)
    val_indices = np.random.permutation(num_episodes_val)

    camera_names = constants.LOW_LEVEL_DATASET_CAMERA_NAMES
    camera_file_suffixes = constants.LOW_LEVEL_DATASET_CAMERA_SUFFIXES
    chunk_size = 60
    use_auto_label = False

    train_dataset = EpisodicDatasetDvrkGeneric(
        train_indices,
        [1],
        "/home/grayson/surpass/srth-surpass/raw_data/Cholecystectomy_grasp_only",
        ["left", "left_wrist", "right_wrist"],
        camera_file_suffixes,
        chunk_size,
        use_auto_label
    )
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=12,
        pin_memory=True, num_workers=12,  persistent_workers=True
    )
    device = utils.resolve_device(str(cfg.device))

    for aug_cfg in cfg.policy.img_aug_cfg.values():
        if "debug_max_calls" in aug_cfg:
            aug_cfg.debug_max_calls = 0

    policy = instantiate(cfg.policy).to(device)
    output_dir = Path("./srth_test_depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    max_saved_samples = 8
    saved_samples = 0

    with torch.no_grad():
        for data in train_dataloader:
            endoscope_img, lw_img, rw_img, current_pose_data, action_data, is_pad, command_text = utils.collect_data(data, device=device)
            image_dict = policy.preprocess_images(endoscope_img, lw_img, rw_img)

            processed_endo = image_dict["endoscope_img"]
            processed_depth = image_dict["depth_img"]

            batch_size = processed_endo.shape[0]
            for batch_idx in range(batch_size):
                if saved_samples >= max_saved_samples:
                    break

                endo_img = _unnormalize_rgb_image(processed_endo[batch_idx])
                depth_img = _depth_to_rgb(processed_depth[batch_idx])
                panel = np.concatenate([endo_img, depth_img], axis=1)
                Image.fromarray(panel).save(output_dir / f"sample_{saved_samples:04d}.png")
                saved_samples += 1

            if saved_samples >= max_saved_samples:
                break

    print(f"Saved {saved_samples} processed endoscope/depth panels to {output_dir}")

if __name__=="__main__":
    main()
