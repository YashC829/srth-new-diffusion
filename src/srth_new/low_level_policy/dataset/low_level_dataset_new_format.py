"""This is taken directly from the original SRTH code. Modifications are minimal.
"""

import json
from typing import List

import numpy as np
import torch
import os
from pathlib import Path
import random

from omegaconf import DictConfig, OmegaConf
import pandas as pd
import cv2

from srth_new.general import constants
from srth_new.general.utils import dataset

import logging
log = logging.getLogger(__name__)

class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        tissue_sample_ids: List[int],
        phases: DictConfig,
        chunk_size=100
        ):

        super(EpisodicDatasetDvrkGeneric).__init__()
        dataset.validate_selected_phases(phases)
        self.episode_dirs, self.ep_counts = dataset.get_episode_directories(
            dataset_dir, tissue_sample_ids, phases
        )
        random.shuffle(self.episode_dirs)

        self.available_cameras = [
            constants.THIRD_PERSON_CAM_NAME, 
            constants.PSM1_WRIST_CAM_NAME, 
            constants.PSM2_WRIST_CAM_NAME
        ]

        self.phases = phases
        self.tissue_sample_ids = tissue_sample_ids
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size

        log.info("\n")
        log.info("-"*100)
        log.info("Dataset Episode Information:\n%s", json.dumps(self.ep_counts, indent=2))
        log.info("-"*100 + "\n")

    def load_camera_specific_csv(self, csv_path):
        """
        Load CSV and split by camera_source for efficient access.
        Returns a dict mapping camera_source to filtered DataFrame.
        
        Handles both old format (no camera_source column) and new format (with camera_source column).
        """
        if not hasattr(self, "camera_csv_cache"):
            self.camera_csv_cache = dict()

        if csv_path in self.camera_csv_cache:
            return self.camera_csv_cache[csv_path]
        
        # Load full CSV
        csv = pd.read_csv(csv_path)
        
        # Check if CSV has camera_source column (new format) or not (old format)
        camera_csvs = {}
        
        if 'camera_source' in csv.columns:
            # New format: Split by camera source
            for camera_source in ['left', 'right', 'psm1', 'psm2']:
                filtered = csv[csv['camera_source'] == camera_source].reset_index(drop=True)
                camera_csvs[camera_source] = filtered
        else:
            # Old format: All cameras share the same timestamps
            # Each camera source gets the full CSV (they were synchronized)
            for camera_source in ['left', 'right', 'psm1', 'psm2']:
                camera_csvs[camera_source] = csv.copy()
        
        # Cache for future use
        self.camera_csv_cache[csv_path] = camera_csvs
        
        return camera_csvs

    def _get_low_level_phase_from_csv_path(self, csv_path: str) -> str:
        return " ".join(Path(csv_path).parts[-3].split("_")[1:])

    def __len__(self):     
        return len(self.episode_dirs)

    def __getitem__(self, index):
        episode_dir = self.episode_dirs[index]
        csv_path = os.path.join(episode_dir, constants.EPISODE_CSV_FILENAME)

        camera_csvs = self.load_camera_specific_csv(csv_path)
        
        # select random camera to serve as timestep anchor for the rest of the
        # data
        selected_camera = np.random.choice(self.available_cameras)
        selected_csv = camera_csvs[selected_camera]
        csv_timestamps = selected_csv['timestamp'].values
        episode_len = len(selected_csv)
        start_idx = np.random.choice(episode_len)
        start_ts = csv_row_idx = start_idx
        target_timestamp = csv_timestamps[csv_row_idx]

        # load images for all cameras
        # Map camera names to their source identifiers and suffixes
        camera_source_map = {
            constants.THIRD_PERSON_CAM_NAME: (
                constants.THIRD_PERSON_CAM_NAME, 
                f'_{constants.THIRD_PERSON_CAM_IMG_SUFFIX}.jpg', 
                constants.THIRD_PERSON_CAM_DIR_NAME
            ),
            constants.PSM2_WRIST_CAM_NAME: (
                constants.PSM2_WRIST_CAM_DIR_NAME, 
                f'_{constants.PSM2_WRIST_CAM_IMG_SUFFIX}.jpg', 
                constants.PSM2_WRIST_CAM_DIR_NAME
            ),
            constants.PSM1_WRIST_CAM_NAME: (
                constants.PSM1_WRIST_CAM_NAME, 
                f'_{constants.PSM1_WRIST_CAM_IMG_SUFFIX}.jpg',
                constants.PSM1_WRIST_CAM_DIR_NAME
            ),
        }

        img_dict = dict()

        for cam_name in self.available_cameras:
            _, suffix, subdir = camera_source_map[cam_name]
            cam_csv = camera_csvs[cam_name]

            time_diffs = np.abs(cam_csv['timestamp'].values - target_timestamp)
            closest_idx = np.argmin(time_diffs)
        
            cam_img_dir = os.path.join(episode_dir, subdir)

            frame_filename = f"frame{closest_idx:06d}{suffix}"
            frame_path = os.path.join(cam_img_dir, frame_filename)
            img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).to(torch.uint8).permute(2, 0, 1)

            if img is None:
                raise Exception(f"No image file {frame_path} exists. Might need to use fallback code from old srth version.")

            img_dict[cam_name] = img_tensor

        # get current position and actions from the selected camera's CSV
        qpos_psm1 = selected_csv[constants.HEADER_NAME_QPOS_PSM1].iloc[start_ts, :].to_numpy()
        action_psm1 = selected_csv[constants.HEADER_NAME_ACTIONS_PSM1].iloc[start_ts:start_ts+self.chunk_size].to_numpy()
        qpos_psm2 = selected_csv[constants.HEADER_NAME_QPOS_PSM2].iloc[start_ts, :].to_numpy()
        action_psm2 = selected_csv[constants.HEADER_NAME_ACTIONS_PSM2].iloc[start_ts:start_ts+self.chunk_size].to_numpy()

        action_len = min(episode_len - start_ts, self.chunk_size)
        padded_action = np.zeros((self.chunk_size, 16), dtype=np.float32)
        padded_action[:action_len] = np.column_stack((action_psm1, action_psm2))
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        current_pose = np.concatenate((qpos_psm1, qpos_psm2)).astype(np.float32)

        # construct observations
        current_pose_data = torch.from_numpy(current_pose).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        command_text = self._get_low_level_phase_from_csv_path(csv_path)

        return (
            img_dict[constants.THIRD_PERSON_CAM_NAME],
            img_dict[constants.PSM2_WRIST_CAM_NAME],
            img_dict[constants.PSM1_WRIST_CAM_NAME],
            current_pose_data,                          # current pose in endo frame
            action_data,                                # action setpoint in endo frame
            is_pad,                                     
            command_text
        )
    
if __name__=="__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
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
                "13_grasper_to_home"
            ],
        }
    )

    dataset = EpisodicDatasetDvrkGeneric(
        dataset_dir="/srv/shared_data/Cholecystectomy_processed/Cholecystectomy",
        tissue_sample_ids=[1, 2, 3, 4, 5],
        phases=phases,
        chunk_size=100
    )
    endo_img, lw_img, rw_img, current_pose_data, action_data, is_pad, command_text = dataset[0]
    
    from PIL import Image
    Image.fromarray(endo_img.cpu().numpy().transpose(1, 2, 0)).save("endo_img.png")
    Image.fromarray(lw_img.cpu().numpy().transpose(1, 2, 0)).save("lw_img.png")
    Image.fromarray(rw_img.cpu().numpy().transpose(1, 2, 0)).save("rw_img.png")