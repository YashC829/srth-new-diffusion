"""This is taken directly from the original SRTH code. Modifications are minimal."""

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


class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        tissue_sample_ids: List[str],
        phases: DictConfig,
        history_chunk_size: int = 0,
        future_chunk_size: int = 100,
    ):

        super(EpisodicDatasetDvrkGeneric).__init__()
        dataset_utils.validate_selected_phases(phases)

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

        return (
            convert(sample["images.endoscope.left"]),
            convert(sample["images.wrist.left"]),
            convert(sample["images.wrist.right"]),
            sample["state"],
            sample["action"],
            sample["action_is_pad"],
            sample["task"],
        )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    phases = OmegaConf.create(
        {
            "unzipping": [
                "1_grabbing_gallbladder_right",
                # "1_grabbing_gallbladder_right_recovery",
                # "2_initial_incision",
                # "2_initial_incision_recovery",
                # "3_hook_to_local_home",
                # "4_hook_tissue",
                # "4_hook_tissue_recovery",
                # "5_cauterize_tissue_right",
                # "6_hook_to_global_home",
                # "7_grasper_to_home",
                # "8_grabbing_gallbladder_left",
                # "8_grabbing_gallbladder_left_recovery",
                # "9_returning_to_initial_incision",
                # "9_returning_to_initial_incision_recovery",
                # "10_cauterize_tissue_left",
                # "11_regrab",
                # "12_hook_to_global_home",
                # "13_grasper_to_home",
            ],
        }
    )

    dataset = EpisodicDatasetDvrkGeneric(
        repo_id="surpass/cholecystectomy_temp",
        tissue_sample_ids=["Tissue#1"],
        phases=phases,
        history_chunk_size=100,
        future_chunk_size=100,
    )
    endo_img, lw_img, rw_img, current_pose_data, action_data, is_pad, command_text = (
        dataset[1]
    )
    # import pdb
    #
    # pdb.set_trace()

    from PIL import Image

    def convert(img):
        return (img * 255).to(torch.uint8)

    Image.fromarray(convert(endo_img).cpu().numpy().transpose(1, 2, 0)).save(
        "endo_img.png"
    )
    Image.fromarray(convert(lw_img).cpu().numpy().transpose(1, 2, 0)).save("lw_img.png")
    Image.fromarray(convert(rw_img).cpu().numpy().transpose(1, 2, 0)).save("rw_img.png")

