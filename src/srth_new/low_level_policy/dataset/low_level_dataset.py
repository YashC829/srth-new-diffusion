"""This is taken directly from the original SRTH code. Modifications are minimal.

# TODO: We still need to audit all of the logic here to understand the code.
"""

import numpy as np
import torch
import os
import random

import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from tqdm import tqdm
import json

from srth_new.low_level_policy.dataset.img_aug import DataAug

import bisect # Required for fast timestamp matching

import IPython
e = IPython.embed

import logging
log = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        tissue_sample_ids, 
        dataset_dir,
        camera_names, 
        camera_file_suffixes,
        chunk_size=100,

        # TODO: There are a few places where use_auto_label significantly changes logic.
        # For now, set to False, but should probably understand a bit more what this does
        # before removing it entirely
        use_auto_label: bool = False,
        ):

        super(EpisodicDatasetDvrkGeneric).__init__()

        self.episode_ids = episode_ids
        self.tissue_sample_ids = tissue_sample_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.use_auto_label = use_auto_label # TODO: Should we remove this?
        self.chunk_size = chunk_size

        self.img_height, self.img_width = [360, 480]
        self.num_samples = len(episode_ids)
        self.arm_command_labels = ["move left arm to the left", "move left arm higher", "move left arm away from me", 
                    "move left arm to the right", "move left arm lower", "move left arm towards me", 
                    "move right arm to the left", "move right arm higher", "move right arm away from me",
                    "move right arm to the right", "move right arm lower", "move right arm towards me",
                    "close both grippers", "close left gripper", "close right gripper",
                    "open both grippers", "open left gripper", "open right gripper",
                    "do not move"]
        # Load the tissue samples and their phases and demos (for later stitching of the episodes)        
        self.tissue_phase_demo_dict = {}
        self.command_text_dict = {}
        # Dictionary to track which episodes are recovery demos (new format)
        self.recovery_episodes = {}

        for tissue_sample_id in tissue_sample_ids:
            tissue_sample_name = f"tissue_{tissue_sample_id}"
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = os.listdir(tissue_sample_dir_path)
            self.tissue_phase_demo_dict[tissue_sample_name] = {}
            self.recovery_episodes[tissue_sample_name] = {}

            for phase_sample in phases:
                demo_samples_path = os.path.join(tissue_sample_dir_path, phase_sample)

                if os.path.isfile(demo_samples_path):
                    continue  # Skip if the tissue sample path is not a directory

                demo_samples = os.listdir(demo_samples_path)

                ## remove corrections folder and JSON files
                demo_samples = [
                    demo_sample for demo_sample in demo_samples 
                    if demo_sample != "Corrections" and not demo_sample.endswith(".json")
                ]

                ## initialize the dictionary for the tissue sample
                if tissue_sample_name not in self.tissue_phase_demo_dict:
                    self.tissue_phase_demo_dict[tissue_sample_name] = {}

                # Load recovery episodes from JSON (new format)
                recovery_json_path = os.path.join(demo_samples_path, "recovery_episodes.json")
                recovery_episode_set = set()
                if os.path.exists(recovery_json_path):
                    try:
                        with open(recovery_json_path, 'r') as f:
                            recovery_data = json.load(f)
                            # Handle different JSON structures
                            if isinstance(recovery_data, list):
                                recovery_episode_set = set(recovery_data)
                            elif isinstance(recovery_data, dict):
                                # Check for 'recovery_episodes' key first (new format)
                                if 'recovery_episodes' in recovery_data:
                                    recovery_episode_set = set(recovery_data['recovery_episodes'])
                                # Fallback to 'episodes' key for backward compatibility
                                elif 'episodes' in recovery_data:
                                    recovery_episode_set = set(recovery_data['episodes'])
                        log.info(f"Loaded {len(recovery_episode_set)} recovery episodes for {tissue_sample_name}/{phase_sample}")
                    except Exception as e:
                        log.info(f"Warning: Failed to load recovery_episodes.json for {phase_sample}: {e}")

                # Store recovery episodes info for this phase
                if phase_sample not in self.recovery_episodes[tissue_sample_name]:
                    self.recovery_episodes[tissue_sample_name][phase_sample] = recovery_episode_set

                # Add or update the demo samples in the dictionary
                self.tissue_phase_demo_dict[tissue_sample_name].setdefault(phase_sample, []).extend(demo_samples)


        log.info(f"num of tissues: {len(self.tissue_phase_demo_dict.keys())}")
        log.info(self.tissue_phase_demo_dict.keys())
        # log.info("phases:", self.tissue_phase_demo_dict[tissue_sample_name].keys())
        temp_info = {phase: len(demo_samples) for phase, demo_samples in self.tissue_phase_demo_dict[tissue_sample_name].items()}
        log.info(f"num of demos per phase: {temp_info}", )
        
        # log.info("num of samples:", sum(len(samples) for samples in self.tissue_phase_demo_dict.values()))
        total_count = 0
        for phase_dict in self.tissue_phase_demo_dict.values():
            for demo_samples in phase_dict.values():
                total_count += len(demo_samples)
        self.num_samples = total_count
        log.info(f"total count: {total_count}")
        # log.info("self.command_embeddings_dict: ", self.command_embeddings_dict.keys())
        unique_phase_folder_names = np.unique(
            [
                phase_folder_name
                for tissue_sample in self.tissue_phase_demo_dict.values()
                for phase_folder_name in tissue_sample.keys()
            ]
        )
        self.command_text_dict = self.get_command_texts(unique_phase_folder_names)
        self.all_samples = [(tissue_sample, phase, sample) 
                            for tissue_sample in self.tissue_phase_demo_dict
                            for phase in self.tissue_phase_demo_dict[tissue_sample]
                            for sample in self.tissue_phase_demo_dict[tissue_sample][phase]]
        
        ## for weighted random sampler
        self.sample_task_labels = []
        for sample in self.all_samples:
            _, phase, _ = sample
            task_label = phase.split("_")[0]  # "1", "2", or "3"
            self.sample_task_labels.append(task_label)

        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]

        self.header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                    "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                    "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                    "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                    "psm2_jaw_sp"]
        
        self.header_ecm = ["ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z",
                            "ecm_pose.orientation.x", "ecm_pose.orientation.y", 
                            "ecm_pose.orientation.z", "ecm_pose.orientation.w"]
        
        self.quat_cp_psm1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
        self.quat_cp_psm2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

        # Load wrist calibration configs per phase
        # These will be used as task-level fallback when episode-level configs don't exist
        self.phase_wrist_configs = {}
        if hasattr(self, 'dataset_dir') and self.dataset_dir:
            for tissue_sample in self.tissue_phase_demo_dict.values():
                for phase_folder_name in tissue_sample.keys():
                    phase_path = None
                    # Find the actual phase directory path
                    for tissue_id in tissue_sample_ids:
                        tissue_name = f"tissue_{tissue_id}"
                        potential_phase_path = os.path.join(self.dataset_dir, tissue_name, phase_folder_name)
                        if os.path.exists(potential_phase_path):
                            phase_path = potential_phase_path
                            break
                    
                    if phase_path:
                        wrist_config_path = os.path.join(phase_path, "wrist_rotation.json")
                        if os.path.exists(wrist_config_path):
                            try:
                                with open(wrist_config_path, 'r') as f:
                                    self.phase_wrist_configs[phase_folder_name] = json.load(f)
                                # log.info(f"Loaded wrist calibration config for phase: {phase_folder_name}")
                            except Exception as e:
                                log.info(f"Warning: Failed to load wrist calibration config for {phase_folder_name}: {e}")
        
        # Pass None for wrist_config since we'll provide it per-episode in __getitem__
        self.transforms = DataAug([self.img_height, self.img_width], use_history=(False), # TODO: Use history hardcoded to False
                                  stereo=False, dataset_dir=self.dataset_dir, wrist_config=None) # TODO: stereo is hardcoded to False
        
        # Dictionary to cache camera-specific CSV data per episode
        self.camera_csv_cache = {}
        
        # Dictionary to cache available image files per camera directory (for fallback when exact timestamp doesn't exist)
        self.image_file_cache = {}
        log.info("\n")


    def is_recovery_episode(self, tissue_sample, phase, sample):
        """
        Check if an episode is a recovery demo.
        Supports both old format (phase ends with '_recovery') and new format (recovery_episodes.json).
        
        Args:
            tissue_sample: Name of the tissue sample
            phase: Phase folder name
            sample: Demo sample name
            
        Returns:
            bool: True if this is a recovery episode
        """
        # Old format: check if phase ends with '_recovery'
        if phase.endswith("_recovery"):
            return True
        
        # New format: check if sample is in recovery_episodes.json
        if tissue_sample in self.recovery_episodes:
            if phase in self.recovery_episodes[tissue_sample]:
                if sample in self.recovery_episodes[tissue_sample][phase]:
                    return True
        
        return False

    def normalize_phase_folder_name(self, phase_folder_name):
        if phase_folder_name.endswith("_recovery"):
            phase_folder_name = phase_folder_name[:-9]
        elif phase_folder_name.startswith("ACTUAL_CUTTING"):
            if phase_folder_name.endswith("_left"):
                phase_folder_name = "8_go_to_the_cutting_position_left_tube"
            elif phase_folder_name.endswith("_right"):
                phase_folder_name = "16_go_to_the_cutting_position_right_tube"
        return phase_folder_name

    def get_command_texts(self, unique_phase_folder_names):
        phase_command_dict = {}

        for phase_folder_name in tqdm(unique_phase_folder_names, desc="Resolving phase commands"):
            normalized_phase_name = self.normalize_phase_folder_name(phase_folder_name)
            _, phase_command = (
                normalized_phase_name.split("_")[0],
                " ".join(normalized_phase_name.split("_")[1:]),
            )
            phase_command_dict[normalized_phase_name] = phase_command

        return phase_command_dict


    def preprocess_img(self, img, start_ts):
        if img is None:
            log.info(f"Image is None: {start_ts}")
            # Return a zero tensor to prevent the batch from crashing
            return torch.zeros((3, self.img_height, self.img_width))

        # 1. Calculate the scaling factor to fit the image into the target box
        # Aspect Ratio Preservation Logic:
        # scale = min(target_w / original_w, target_h / original_h)
        h_orig, w_orig = img.shape[:2]
        scale = min(self.img_width / w_orig, self.img_height / h_orig)
        
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)

        # 2. Resize maintaining the aspect ratio
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. Calculate padding to center the image on the target canvas
        pad_w = self.img_width - new_w
        pad_h = self.img_height - new_h
        
        # Distribute padding evenly (left/right, top/bottom)
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)

        # 4. Apply padding (black bars)
        img_final = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # 5. Convert Color and Type
        img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

        # Convert to tensor and change HWC -> CHW 
        # (Using .permute is slightly more standard for this than einsum)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1)

        # Normalize to [0, 1]
        return img_tensor / 255.0

    def create_offset_map_with_gradient(self, image_shape, insert_point, exit_point, normalize_size=224.0, device='cpu', eps=1e-6):
        """
        Returns a 3-channel offset map:
        - Channel 0: dx to insertion point
        - Channel 1: dy to insertion point
        - Channel 2: scalar heatmap (1 at insertion, 0 at exit)

        Args:
            image_shape: (H, W)
            insert_point: (x, y)
            exit_point: (x, y)
            normalize_size: reference image size for normalization
            device: 'cpu' or 'cuda'
        """
        H, W = image_shape
        normalizing_constant = 250.0 * (min(H, W) / normalize_size)

        y_coords = torch.arange(H, device=device)
        x_coords = torch.arange(W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Offsets to insertion point (dy, dx)
        dx = (x_grid - insert_point[0]) / normalizing_constant
        dy = (y_grid - insert_point[1]) / normalizing_constant

        # Gradient heatmap: insertion → 1.0, exit → 0.0
        d_insert = torch.sqrt((x_grid - insert_point[0]) ** 2 + (y_grid - insert_point[1]) ** 2)
        d_exit = torch.sqrt((x_grid - exit_point[0]) ** 2 + (y_grid - exit_point[1]) ** 2)
        heat = d_exit / (d_insert + d_exit + eps)  # in [0, 1]

        # Stack to shape (3, H, W)
        offset_map = torch.stack([dx, dy, heat], dim=0)
        return offset_map.clamp(-1.0, 1.0)  # Optional clamp


    def offset_map_to_rgb_visual(self, offset_map):
        """
        Converts a (3, H, W) offset map (dx, dy, heat) to a uint8 RGB image for visualization.
        - Red = dx
        - Green = dy
        - Blue = heat
        """
        if torch.is_tensor(offset_map):
            offset_map = offset_map.detach().cpu().numpy()

        # Normalize each channel to [0, 1]
        def normalize(x):
            x = x - np.min(x)
            x = x / (np.max(x) + 1e-6)
            return x

        dx_norm = normalize(offset_map[0])
        dy_norm = normalize(offset_map[1])
        heat_norm = normalize(offset_map[2])

        rgb_image = np.stack([
            dx_norm,     # R
            dy_norm,     # G
            heat_norm    # B
        ], axis=-1)  # (H, W, 3)

        rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        return rgb_uint8


    def find_closest_available_image(self, target_timestamp, camera_dir, suffix):
        """
        Find the closest available image file to the target timestamp.
        This is used as a fallback when the exact timestamp from CSV doesn't have a corresponding image file.
        
        Args:
            target_timestamp: Target timestamp (int or str) or index
            camera_dir: Directory containing image files
            suffix: Image file suffix (e.g., '_psm1.jpg')
            
        Returns:
            Tuple of (image_filename, actual_timestamp) or (None, None) if no images found
        """
        # Create cache key
        cache_key = (camera_dir, suffix)
        
        # Initialize cache if needed
        if cache_key not in self.image_file_cache:
            if not os.path.exists(camera_dir):
                return None, None
                
            # Get all files with the matching suffix
            files = [f for f in os.listdir(camera_dir) if f.endswith(suffix)]
            if not files:
                return None, None
            
            # Check if files are index-based (frameXXXXXX) or timestamp-based
            is_frame_indexed = any(f.startswith('frame') for f in files)
            
            if is_frame_indexed:
                # Old format: frame000000_left.jpg
                # Extract frame indices
                ts_list = []
                for f in files:
                    try:
                        # Extract frame number from "frameXXXXXX_left.jpg"
                        if f.startswith('frame'):
                            frame_num_str = f.replace('frame', '').replace(suffix, '')
                            frame_num = int(frame_num_str)
                            ts_list.append((frame_num, f))
                    except (ValueError, AttributeError):
                        continue
            else:
                # New format: timestamp-based filenames
                # Extract timestamps from filenames
                # Format: {timestamp}{suffix} (e.g., "1768850630120002929_psm1.jpg")
                ts_list = []
                for f in files:
                    try:
                        # Remove suffix and extract timestamp
                        ts_str = f.replace(suffix, "")
                        ts_val = int(ts_str)
                        ts_list.append((ts_val, f))
                    except (ValueError, AttributeError):
                        continue
            
            if not ts_list:
                return None, None
            
            # Sort by timestamp/index for binary search
            ts_list.sort(key=lambda x: x[0])
            self.image_file_cache[cache_key] = (ts_list, is_frame_indexed)
        
        cached_data = self.image_file_cache[cache_key]
        if isinstance(cached_data, tuple):
            ts_list, is_frame_indexed = cached_data
        else:
            # Old cache format, rebuild
            ts_list = cached_data
            is_frame_indexed = False
        
        if not ts_list:
            return None, None
        
        # Convert target_timestamp to int if it's a string
        try:
            target_ts = int(target_timestamp)
        except (ValueError, TypeError):
            return None, None
        
        # Binary search to find the closest timestamp/index
        keys = [x[0] for x in ts_list]
        pos = bisect.bisect_left(keys, target_ts)
        
        if pos == 0:
            return ts_list[0][1], ts_list[0][0]
        if pos == len(ts_list):
            return ts_list[-1][1], ts_list[-1][0]
        
        # Check if the previous one or the current one is closer
        before = ts_list[pos - 1]
        after = ts_list[pos]
        if after[0] - target_ts < target_ts - before[0]:
            return after[1], after[0]
        else:
            return before[1], before[0]

    def load_camera_specific_csv(self, csv_path, episode_key):
        """
        Load CSV and split by camera_source for efficient access.
        Returns a dict mapping camera_source to filtered DataFrame.
        
        Handles both old format (no camera_source column) and new format (with camera_source column).
        """
        if episode_key in self.camera_csv_cache:
            return self.camera_csv_cache[episode_key]
        
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
        self.camera_csv_cache[episode_key] = camera_csvs
        
        return camera_csvs

    def __len__(self):     
        return len(self.episode_ids)

    def __getitem__(self, index):
        # Retry mechanism: try up to 5 times with different random start_ts if images can't be found
        max_retries = 5
        
        for retry_attempt in range(max_retries):
            try:
                
                # Get the tissue sample, phase, and sample based on the index
                episode_id = self.episode_ids[index]
                if episode_id < self.num_samples:
                    tissue_sample, phase, sample = self.all_samples[episode_id]
                else:
                    log.info("episode_id out of range")
                    tissue_sample, phase, sample = self.all_samples[episode_id % self.num_samples]

                # Ensure dataset_path points to the demo directory, not a file
                dataset_path = os.path.join(self.dataset_dir, tissue_sample, phase, sample)
                
                # Verify it's actually a directory
                if not os.path.isdir(dataset_path):
                    raise ValueError(f"Expected directory but got: {dataset_path}")
                
                csv_path = os.path.join(dataset_path, "ee_csv.csv")
                
                # Create episode key for caching
                episode_key = f"{tissue_sample}/{phase}/{sample}"
                
                # Load camera-specific CSV data (with caching)
                camera_csvs = self.load_camera_specific_csv(csv_path, episode_key)
                
                # Randomly select a camera to sample from
                available_cameras = [cam for cam in ['left', 'right', 'psm1', 'psm2'] 
                                   if len(camera_csvs[cam]) > 0]
                
                if not available_cameras:
                    raise ValueError(f"No camera data available in {csv_path}")
                
                # Select random camera for this sample
                selected_camera = np.random.choice(available_cameras)
                selected_csv = camera_csvs[selected_camera]
                
                # Check if CSV has timestamp column (new format with camera_source)
                has_timestamp = 'timestamp' in selected_csv.columns
                
                if has_timestamp:
                    csv_timestamps = selected_csv['timestamp'].values
                else:
                    # Old format: use row indices as "timestamps"
                    csv_timestamps = np.arange(len(selected_csv))
                
                episode_len = len(selected_csv)
                start_idx = np.random.choice(episode_len)
                start_ts = start_idx
                
                csv_row_idx = start_idx

                # Get the target timestamp from the selected camera's CSV
                target_timestamp = csv_timestamps[csv_row_idx]

                # -------------------------------
                # 1. Load images for all cameras
                # -------------------------------
                # Map camera names to their source identifiers and suffixes
                camera_source_map = {
                    'left': ('left', '_left.jpg', 'left_img_dir'),
                    'right': ('right', '_right.jpg', 'right_img_dir'),
                    'left_wrist': ('psm1', '_psm1.jpg', 'endo_psm1'),
                    'right_wrist': ('psm2', '_psm2.jpg', 'endo_psm2'),
                }
                
                img_dict_raw = {}
                for cam_name in self.camera_names:
                    source, suffix, subdir = camera_source_map[cam_name]
                    
                    # Get the CSV for this camera source
                    cam_csv = camera_csvs[source]
                    
                    if len(cam_csv) == 0:
                        raise FileNotFoundError(f"No {source} camera data in CSV")
                    
                    # Find the row with timestamp closest to our target
                    if has_timestamp:
                        # New format: Use actual timestamps for matching
                        time_diffs = np.abs(cam_csv['timestamp'].values - target_timestamp)
                        closest_idx = np.argmin(time_diffs)
                        image_timestamp = cam_csv['timestamp'].iloc[closest_idx]
                    else:
                        # Old format: Use row index directly (all cameras synchronized)
                        closest_idx = csv_row_idx if csv_row_idx < len(cam_csv) else len(cam_csv) - 1
                        # For old format, we need to get the actual timestamp from the image filename
                        # or construct it from row index - we'll use find_closest_available_image
                        image_timestamp = target_timestamp  # This is just the row index
                    
                    # Construct image filename (format: {timestamp}{suffix}, e.g., "1768850630120002929_left.jpg")
                    camera_dir = os.path.join(dataset_path, subdir)
                    
                    # Check if images in this directory are frame-indexed or timestamp-based
                    # This is important for datasets with timestamp CSV but frame-indexed images
                    if not os.path.exists(camera_dir):
                        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
                    
                    sample_files = [f for f in os.listdir(camera_dir) if f.endswith(suffix)][:5]
                    images_are_frame_indexed = any(f.startswith('frame') for f in sample_files) if sample_files else False
                    
                    if has_timestamp and not images_are_frame_indexed:
                        # Case 1: CSV has timestamps, images have timestamps
                        # New format: timestamp is actual nanosecond timestamp
                        image_filename = f"{image_timestamp}{suffix}"
                        image_path = os.path.join(camera_dir, image_filename)
                        
                        # Try to load image with exact timestamp
                        img = cv2.imread(image_path)
                        
                        # If image not found, try to find the closest available image file
                        if img is None:
                            fallback_filename, fallback_timestamp = self.find_closest_available_image(
                                image_timestamp, camera_dir, suffix
                            )
                            if fallback_filename is not None:
                                fallback_path = os.path.join(camera_dir, fallback_filename)
                                img = cv2.imread(fallback_path)
                                if img is not None:
                                    # Use the fallback image (with a warning if timestamp difference is large)
                                    time_diff = abs(int(fallback_timestamp) - int(image_timestamp))
                                    if time_diff > 100000000:  # More than 100ms difference (in nanoseconds)
                                        log.info(f"Warning: Using fallback image for {cam_name}. "
                                              f"Requested timestamp: {image_timestamp}, "
                                              f"Found timestamp: {fallback_timestamp}, "
                                              f"Difference: {time_diff} ns")
                                else:
                                    raise FileNotFoundError(
                                        f"Image not found at: {image_path} "
                                        f"and fallback image also failed: {fallback_path}"
                                    )
                            else:
                                raise FileNotFoundError(
                                    f"Image not found at: {image_path} "
                                    f"and no images found in directory: {camera_dir}"
                                )
                    else:
                        # Case 2: Either CSV has no timestamps OR images are frame-indexed
                        # Use CSV row index to find corresponding frame image
                        # Try direct frame filename first
                        frame_filename = f"frame{closest_idx:06d}{suffix}"
                        frame_path = os.path.join(camera_dir, frame_filename)
                        img = cv2.imread(frame_path)
                        
                        if img is None:
                            # Fallback: find closest available frame by index
                            fallback_filename, fallback_idx = self.find_closest_available_image(
                                closest_idx, camera_dir, suffix
                            )
                            if fallback_filename is not None:
                                fallback_path = os.path.join(camera_dir, fallback_filename)
                                img = cv2.imread(fallback_path)
                                if img is None:
                                    raise FileNotFoundError(
                                        f"Image not found at: {frame_path} "
                                        f"and fallback also failed: {fallback_path} "
                                        f"for camera {cam_name} in directory: {camera_dir}"
                                    )
                            else:
                                raise FileNotFoundError(
                                    f"No images found for camera {cam_name} at index {closest_idx} "
                                    f"in directory: {camera_dir}"
                                )
                    
                    img_dict_raw[cam_name] = img

                # -------------------------------
                # 3. Preprocess and augment images
                # -------------------------------
                img_dict = {k: self.preprocess_img(v, start_ts) for k, v in img_dict_raw.items()}


                # -------------------------------
                #  5. Apply data augmentation
                # -------------------------------
                
                # Get phase-level wrist config as fallback
                phase_config = self.phase_wrist_configs.get(phase, None)
                
                tfmed = self.transforms(img_dict, episode_path=dataset_path, phase_config=phase_config)
                
                # Stack and convert to float tensor in [0, 1] range
                # DataAug returns uint8 [0, 255], but policy expects float [0, 1]
                image_data = np.stack([tfmed[k] for k in sorted(tfmed.keys())], axis=0)
                image_data = torch.from_numpy(image_data).float() / 255.0


                # -------------------------------
                #  6. Load and compute action data from selected camera
                # -------------------------------

                # get current position and actions from the selected camera's CSV
                qpos_psm1 = selected_csv[self.header_name_actions_psm1].iloc[start_ts, :].to_numpy()
                action_psm1 = selected_csv[self.header_name_actions_psm1].iloc[start_ts:start_ts+400].to_numpy()
                qpos_psm2 = selected_csv[self.header_name_actions_psm2].iloc[start_ts, :].to_numpy()
                action_psm2 = selected_csv[self.header_name_actions_psm2].iloc[start_ts:start_ts+400].to_numpy()

                action_len = min(episode_len - start_ts, 400)
                padded_action = np.zeros((400, 16), dtype=np.float32)
                padded_action[:action_len] = np.column_stack((action_psm1, action_psm2))
                is_pad = np.zeros(400)
                is_pad[action_len:] = 1

                current_pose = np.concatenate((qpos_psm1, qpos_psm2)).astype(np.float32)

                # construct observations
                current_pose_data = torch.from_numpy(current_pose).float()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()

                # -------------------------------
                #  7. Command Text
                # -------------------------------
                directional_label = None

                is_recovery = self.is_recovery_episode(tissue_sample, phase, sample)
                base_phase = self.normalize_phase_folder_name(phase)

                if is_recovery:
                    directional_label = get_auto_label(selected_csv, start_ts)

                phase_command = self.command_text_dict.get(base_phase)
                if phase_command is None:
                    raise ValueError(f"Phase '{base_phase}' not found in command_text_dict")

                if self.use_auto_label and directional_label is not None and directional_label != "do not move":
                    command_text = directional_label
                else:
                    command_text = phase_command

                return image_data, current_pose_data, action_data, is_pad, command_text
            
            # Handle FileNotFoundError during image loading - retry with different start_ts
            except FileNotFoundError as e:
                # Check if this is an image loading error (should retry)
                error_msg = str(e)
                if "Image not found" in error_msg or "no images found" in error_msg.lower():
                    if retry_attempt < max_retries - 1:
                        # Retry with a new random start_ts
                        continue
                    else:
                        # All retries exhausted, raise the error
                        log.info(f"File not found at index {index} after {max_retries} retries: {e}")
                        raise
                else:
                    # Other FileNotFoundError (e.g., CSV file not found) - don't retry
                    log.info(f"File not found at index {index}: {e}")
                    raise
            
            # Handle other exceptions - don't retry
            except pd.errors.EmptyDataError as e:
                log.info(f"Empty data error at index {index}: {e}")
                raise
            except KeyError as e:
                log.info(f"Key error at index {index}: {e}")
                raise
            except ValueError as e:
                log.info(f"Value error at index {index}: {e}")
                raise
            except Exception as e:
                log.info(f"Unexpected error at index {index}: {e}")
                raise  # MUST re-raise to prevent returning None!
        
        # This should never be reached, but add safety check just in case
        raise RuntimeError(f"Failed to load data at index {index} after {max_retries} retries without raising an exception")

        
"""
Test the EpisodicDatasetDvrkGeneric class.
"""
if __name__ == "__main__":
    # seed = random.randint(0, 1000)
    set_seed(0)
    # seed = random.randint(0, 1000)
    # set_seed(seed)
    # Parameters for the test
    path_to_dataset = os.getenv("PATH_TO_DATASET")
    # path_to_dataset = "/home/imerse/chole_ws/data"

    dataset_dir = os.path.join(path_to_dataset, "cnh_exvivo_chole")
    from dvrk_scripts.constants_dvrk import TASK_CONFIGS
    task_config = TASK_CONFIGS['cnh_exvivo_chole_4_mono']
    camera_names = task_config['camera_names']
    tissue_samples_ids = task_config["tissue_samples_ids"]
    num_episodes = task_config["num_episodes"]
    camera_file_suffixes = task_config['camera_file_suffixes']
    episode_ids = [i for i in range(num_episodes)]
    no_qpos = task_config.get('no_qpos', False)
    log.info("no_qpos:", no_qpos)
    dataset = EpisodicDatasetDvrkGeneric(
                episode_ids,
                tissue_samples_ids,
                dataset_dir,
                camera_names,
                camera_file_suffixes,
                chunk_size=60,
                )
    for i in range(10):

        # Sample a random item from the dataset
        rdm_idx = np.random.randint(0, len(dataset))
        log.info("idx:", rdm_idx)
        image_data, current_pose_data, action_data, is_pad, command_text = dataset[rdm_idx]


        # Create a figure with subplots: one row per timestamp, one column per camera
        
        fig, axes = plt.subplots(1, len(image_data), figsize=(15, 10))
        
        # Handle the case when there's only one camera (axes is not an array)
        if len(image_data) == 1:
            axes = [axes]
        
        for cam_idx, img in enumerate(image_data):

            # Check and possibly transpose the shape if needed
            if img.shape[0] == 3 and len(img.shape) == 3:
                img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

            axes[cam_idx].imshow(img)
            axes[cam_idx].axis('off')  # Optionally turn off the axis

        # set the title of the figure
        # fig.suptitle(f"{command}")
        plt.show()
        plt.savefig(f"./visualization_{i}.png")
        
        # fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        # for cam_idx, cam_name in enumerate(camera_names):
        #     img = image_data[cam_idx]  # Assuming image_data is a numpy array or compatible type

        # # Check and possibly transpose the shape if needed
        # if img.shape[0] == 3 and len(img.shape) == 3:
        #     img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

        # axes.imshow(img)
        # axes.set_title("left_img")
        # axes.axis('off')  # Optionally turn off the axis
        # plt.show()
        # plt.savefig(f"./visualization_{i}.png")
