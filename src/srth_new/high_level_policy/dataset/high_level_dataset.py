"""This code was adapted from the original SRT-H's SequenceDataset. We cleaned
up several extraneous experimental arguments and simplified the dataset. Original
dataset code found here: 
https://github.com/gbyrd-research/srth-surpass/blob/7695d60ea7f117cd316e3fb85a3b345da532d5a7/src/instructor/dataset_daVinci.py

TODO: Format usage notes more

Dataset must strictly conform to the following format:
    $PATH_TO_DATASET
    ├── [DATASET_NAME]       # the dataset base dir
    |   └── tissue_1                      # data subset
    |   |   ├── 1_[task_name]             # task name
    |   |   |   ├── [episode]             # should be timestamp when the data was recorded
    |   |   |   |      ├── left_img_dir   # left endoscope cam images (frame000000_left.jpg)
    |   |   |   |      ├── right_img_dir  # right endoscope cam images (frame000000_right.jpg)
    |   |   |   |      ├── endo_psm1      # right wrist cam images (frame000000_psm1.jpg)
    |   |   |   |      ├── endo_psm2      # left wrist cam images (frame000000_psm2.jpg)
    |   |   |   |      └── ee_csv.csv     # kinematics
    |   └── tissue_2                      # data subset
    |   |   ├── 1_[task_name]             # task name
    |   |   |   ├── [episode]             # should be timestamp when the data was recorded
    |   |   |   |      ├── left_img_dir   # left endoscope cam images (frame000000_left.jpg)
    |   |   |   |      ├── right_img_dir  # right endoscope cam images (frame000000_right.jpg)
    |   |   |   |      ├── endo_psm1      # right wrist cam images (frame000000_psm1.jpg)
    |   |   |   |      ├── endo_psm2      # left wrist cam images (frame000000_psm2.jpg)
    |   |   |   |      └── ee_csv.csv     # kinematics
    ...

"""

import os
from pathlib import Path
from typing import List, Literal

from omegaconf import DictConfig
import numpy as np
import torch
from torchvision import transforms

from srth_new.general import utils
from srth_new.general import constants

class HighLevelDatasetSRTH(torch.utils.data.Dataset):
    def __init__(
        self,
        split_name: Literal["train", "val"],
        metadata_cfg: DictConfig,
        temporal_sampling_cfg: DictConfig,
        recovery_cfg: DictConfig,
        patch_extraction_cfg: DictConfig,
        multitask_labels: List[str]
    ):
        super().__init__(self)

        self.split_name = split_name
        self.metadata_cfg = metadata_cfg
        self.temporal_sampling_cfg = temporal_sampling_cfg
        self.recovery_cfg = recovery_cfg
        self.patch_extraction_cfg = patch_extraction_cfg
        self.multitask_labels = multitask_labels

        self._load_dataset_info(metadata_cfg)

        self.resize = transforms.Resize((constants.IMG_RESIZE_SIZE), antialias=True)

    def _load_dataset_info(self, cfg: DictConfig) -> None:
        """Loads the information required to sample from the dataset in the 
        __getitem__ function."""
        if self.split_name == "train":
            self.tissue_ids = cfg.train_tissue_ids
        elif self.split_name == "val":
            self.tissue_ids = cfg.val_tissue_ids

        # get phase metadata dictionary. this provides a dictionary where the high
        # level keys are phase names and the values are lists of dictionaries that
        # contain the episode folder name and number of frames in that phase
        # training trajectory
        uq_phase_dir_names = set()
        phase_info_dict = dict()
        for tissue_id in self.tissue_ids:
            tissue_dir = Path(cfg.dataset_dir).joinpath(constants.TISSUE_FOLDER_NAME(tissue_id))
            phases = utils.get_sorted_phases(tissue_dir)
            [uq_phase_dir_names.add(x) for x in phases]
            for phase_dir in uq_phase_dir_names:
                phase_info_dict[phase_dir] = list()
                episode_dir_names = os.listdir(Path(tissue_dir).joinpath(tissue_dir))
                # add episode names and len to the phase dict
                for ep_dir_name in episode_dir_names:
                    ep_dir_path = tissue_dir.joinpath(phase_dir, ep_dir_name)
                    num_phase_frames = len(os.listdir(ep_dir_path.joinpath(constants.THIRD_PERSON_CAM_DIR_NAME)))
                    phase_info_dict[phase_dir].append(
                        {"episode_dir": ep_dir_name, "num_phase_frames": num_phase_frames}
                    )

        # Generate the embeddings for all phase commands
        encoder_name = "distilbert"
        tokenizer, model = utils.initialize_model_and_tokenizer(encoder_name)
        self.command_embeddings_dict = utils.generate_command_embeddings(
            uq_phase_dir_names, encoder_name, tokenizer, model
        ) 
        del tokenizer, model

        # Compute phase len stats for each phase in the dataset
        self.phase_len_stats = dict()
        for phase, info_list in phase_info_dict.items():
            self.phase_len_stats[phase] = dict()
            phase_lens = [x["num_phase_frames"] for x in info_list]
            self.phase_len_stats[phase] = {
                "min": min(phase_lens),
                "max": max(phase_lens),
                "mean": sum(phase_lens) / len(phase_lens),
                "std": np.std(phase_lens),
                "num_demos": len(phase_lens),
            }

        # Get camera patch names
        camera_names = constants.CAMERA_NAMES
        camera_patch_names = []
        for camera_name in camera_names:
            if camera_name in ["endo_psm2", "endo_psm1"]:
                camera_patch_names.append(camera_name)
            else:
                camera_patch_names.append(f"{camera_name}_global")
                camera_patch_names.append(f"{camera_name}_center")
        self.camera_patch_names = camera_patch_names

    def __getitem__(self, index):
        
        # select a random tissue sample to generate the episode from
        tissue_sample = constants.TISSUE_FOLDER_NAME(np.random.choice(self.tissue_ids))
        
