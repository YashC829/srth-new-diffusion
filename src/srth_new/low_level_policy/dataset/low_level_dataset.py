"""This is taken directly from the original SRTH code. Modifications are minimal."""

import shutil
import logging
from typing import List

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from srth_new.general import constants
from srth_new.general.utils import dataset as dataset_utils

log = logging.getLogger(__name__)


def get_high_level_phases_from_phase_cfg(cfg: DictConfig):
    high_level_phases = list()
    for high_level_phase, _ in cfg.items():
        high_level_phases.append(high_level_phase)
    return high_level_phases


def get_low_level_phases_from_phase_cfg(cfg: DictConfig):
    low_level_phases = list()
    for _, low_level_phase_list in cfg.items():
        low_level_phases.extend(low_level_phase_list)
    return low_level_phases


class FilteredLeRobotDataset(Dataset):
    def __init__(self, lerobot_ds, indices):
        self.lerobot_ds = lerobot_ds
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.lerobot_ds[self.indices[i]]

    def __getattr__(self, name):
        # Delegate delta_timesteps, features, fps, meta, etc.
        return getattr(self.lerobot_ds, name)


class DvrkLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        tissue_sample_ids: List[str],
        phases: DictConfig,
        history_chunk_size: int = 0,
        future_chunk_size: int = 100,
    ):

        super(DvrkLerobotDataset).__init__()
        dataset_utils.validate_selected_phases(phases)
        self.history_chunk_size = history_chunk_size
        self.future_chunk_size = future_chunk_size

        self.ds_meta = LeRobotDatasetMetadata(repo_id)

        delta_timestamps = {
            "action": [
                t / self.ds_meta.fps
                for t in range(-history_chunk_size, future_chunk_size)
            ],
        }

        ds_lerobot = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

        # filter by phase and tissue id
        high_level_phases = [
            dataset_utils.clean_high_level_phase_from_cfg(x)
            for x in get_high_level_phases_from_phase_cfg(phases)
        ]
        low_level_phases = [
            dataset_utils.clean_low_level_phase_from_cfg(x)
            for x in get_low_level_phases_from_phase_cfg(phases)
        ]
        df = ds_lerobot.hf_dataset.to_pandas()
        mask = (
            df["meta.low_level_phase"].isin(low_level_phases)
            & df["meta.high_level_phase"].isin(high_level_phases)
            & df["meta.tissue_id"].isin(tissue_sample_ids)
        )
        filtered_indices = df.index[mask].tolist()

        self.dataset = FilteredLeRobotDataset(ds_lerobot, filtered_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        def convert(img):
            return (img * 255).to(torch.uint8)

        endoscope_img = convert(sample["images.endoscope.left"])
        lw_img = convert(sample["images.wrist.left"])
        rw_img = convert(sample["images.wrist.right"])
        state = sample["state"]
        action_history_is_pad = torch.empty((0,), dtype=torch.bool)
        action = sample["action"][self.history_chunk_size:]
        action_is_pad = sample["action_is_pad"][self.history_chunk_size:]
        command_text = sample["task"]

        tool_kp = sample["tool_kp"] if "tool_kp" in sample else None
        affordance_kp = sample["affordance_kp"] if "affordance_kp" in sample else None 

        # handle the action history
        if self.history_chunk_size > 0:
            action_history = sample["action"][:self.history_chunk_size]
            action_history_is_pad = sample["action_is_pad"][:self.history_chunk_size]
        else:
            action_dim = sample["action"].shape[-1]
            action_history = torch.empty((0, action_dim), dtype=sample["action"].dtype)

        return (
            endoscope_img,
            lw_img,
            rw_img,
            state,
            action_history,
            action_history_is_pad,
            action,
            action_is_pad,
            command_text,
            affordance_kp,
            tool_kp,
            sample["original_ep_dir"]
        )


if __name__ == "__main__":

    from srth_new.low_level_policy import utils
    from PIL import Image, ImageDraw
    from pathlib import Path
    import os

    debug_dir = Path("./debug_dataset_annotations")
    os.makedirs(debug_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    phases = OmegaConf.create(
        {
            "unzipping": [
                "1_grabbing_gallbladder_right",
                "1_grabbing_gallbladder_right_recovery",
                "2_initial_incision",
                "2_initial_incision_recovery",
                "3_hook_to_local_home",
                "4_hook_tissue",
                "4_hook_tissue_recovery",
                "5_cauterize_tissue_right",
                "6_hook_to_global_home",
                "7_grasper_to_home",
                "8_grabbing_gallbladder_left",
                "8_grabbing_gallbladder_left_recovery",
                "9_returning_to_initial_incision",
                "9_returning_to_initial_incision_recovery",
                "10_cauterize_tissue_left",
                "11_regrab",
                "12_hook_to_global_home",
                "13_grasper_to_home",
            ],
        }
    )

    dataset = DvrkLerobotDataset(
        repo_id="surpass/cholecystectomy_debug",
        tissue_sample_ids=[
            "Tissue#1", "Tissue#2", "Tissue#3", "Tissue#4", "Tissue#5",
            "Tissue#6", "Tissue#7", "Tissue#8", "Tissue#9", "Tissue#10", "Tissue#11",
        ],
        phases=phases,
        history_chunk_size=0,
        future_chunk_size=100,
    )

    # visualize annotated data
    num_samples = 1000
    for i in range(num_samples):
        sample = dataset[i]
        data = utils.collect_data(sample, device=torch.device("cpu"))

        endo_img = data["endoscope_img"]
        tool_kp = data["tool_kp"]
        affordance_kp = data["affordance_kp"]

        # Convert CHW -> HWC for PIL
        img_np = endo_img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)

        def draw_kp(kp, color, label):
            x, y = kp.tolist()
            x, y = int(round(x)), int(round(y))

            r = 6
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)
            draw.text((x + 8, y - 8), label, fill=color)

        draw_kp(tool_kp, "red", "tool")
        draw_kp(affordance_kp, "yellow", "aff")

        out_path = debug_dir / f"{sample[-1]}_{i:04d}.png".replace("/", "_")
        
        img.save(out_path)

