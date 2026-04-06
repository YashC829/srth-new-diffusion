import numpy as np
import torch
import os

import cv2
import torchvision.transforms as T
import albumentations as A

import json

import IPython
e = IPython.embed

class DataAug(object):
    def __init__(self, img_hw, use_history, stereo=False, mask_prob=0.07, mask_sketch_prob=0.5, dataset_dir=None, wrist_config=None):
        self.img_hw = img_hw  # (H, W)
        self.ratio = 0.95
        self.mask_prob = mask_prob
        self.use_history = use_history
        self.stereo = stereo
        self.mask_sketch_prob = mask_sketch_prob
        self.dataset_dir = dataset_dir
        
        # Store task-level wrist calibration config as fallback
        # Can be passed directly or will be None (backward compatible)
        self.task_wrist_config = wrist_config
        if wrist_config is not None:
            print(f"Loaded task-level wrist calibration config with keys: {list(wrist_config.keys())}")

        # Spatial transforms (crop, resize, rotation), synced across 'left' and 'mask'
        self.spatial_transforms = A.Compose([
            A.RandomCrop(height=int(img_hw[0] * self.ratio), width=int(img_hw[1] * self.ratio)),
            A.Resize(height=img_hw[0], width=img_hw[1]),
            A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101),
        ], additional_targets={'mask': 'image'})

        # Color jitter (only for RGB)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.08)

        # Albumentations for pixel dropout
        min_height = max(1, img_hw[0] // 40)
        min_width = max(1, img_hw[1] // 40)
        max_height = min(img_hw[0] // 30, img_hw[0])
        max_width = min(img_hw[1] // 30, img_hw[1])

        self.pixel_dropout = A.Compose([
            A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width,
                            min_holes=1, min_height=min_height, min_width=min_width,
                            fill_value=0, p=0.8),
        ], additional_targets={'mask': 'image'})
    
    def load_episode_wrist_rotation(self, episode_path, phase_config=None):
        """
        Load wrist calibration config for a specific episode.
        Falls back to phase-level config if episode-level doesn't exist.
        
        Args:
            episode_path: Path to the episode directory
            phase_config: Phase-level config to use as fallback (default: None)
            
        Returns:
            dict: Calibration config or None
        """
        # Try episode-level first
        episode_rotation_path = os.path.join(episode_path, "wrist_rotation.json")
        if os.path.exists(episode_rotation_path):
            try:
                with open(episode_rotation_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load episode-level wrist calibration config: {e}")
        
        # Fall back to phase-level config
        if phase_config is not None:
            return phase_config
        
        # Final fallback to task-level (if set during init)
        return self.task_wrist_config
    
    def apply_wrist_augmentation(self, img_np, aug_config, camera_key):
        """
        Apply calibration-based augmentation from config to a wrist camera image.
        Supports rotation, brightness, contrast, saturation, gamma, and RGB multipliers.
        
        Args:
            img_np: numpy array (H, W, C), uint8 in [0, 255] range
            aug_config: dict with augmentation parameters per camera (psm1/psm2)
            camera_key: 'left_wrist' or 'right_wrist'
            
        Returns:
            augmented numpy array (H, W, C), uint8
        """
        if aug_config is None:
            return img_np
        
        # Debug: validate aug_config type
        if not isinstance(aug_config, dict):
            print(f"ERROR: aug_config is not a dict! type={type(aug_config)}, value={aug_config}, camera_key={camera_key}")
            return img_np
        
        # Map camera keys to config keys (left_wrist=psm1, right_wrist=psm2)
        config_key = 'psm1' if camera_key == 'left_wrist' else 'psm2'
        
        # Handle both old format (float) and new format (dict)
        cam_config_raw = aug_config.get(config_key, {})
        
        # If old format (just a float rotation value), convert to new format
        if isinstance(cam_config_raw, (int, float)):
            cam_config = {
                'rotation': float(cam_config_raw),
                'brightness': 0.0,
                'contrast': 1.0,
                'saturation': 1.0,
                'gamma': 1.0,
                'r_mul': 1.0,
                'g_mul': 1.0,
                'b_mul': 1.0
            }
        elif isinstance(cam_config_raw, dict):
            cam_config = cam_config_raw
        else:
            print(f"WARNING: Unexpected cam_config type for {config_key}: {type(cam_config_raw)}")
            return img_np
        
        if not cam_config:
            return img_np
        
        # Convert to float32 for processing (keep in [0, 255] range initially)
        img = img_np.astype(np.float32)
        
        # 1. Apply rotation
        rotation = cam_config.get('rotation', 0)
        if rotation != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        
        # 2. Apply brightness (additive)
        brightness = cam_config.get('brightness', 0)
        if brightness != 0:
            img = np.clip(img + brightness, 0, 255)
        
        # 3. Apply contrast (multiplicative around 128)
        contrast = cam_config.get('contrast', 1.0)
        if contrast != 1.0:
            img = np.clip((img - 128) * contrast + 128, 0, 255)
        
        # 4. Convert to [0, 1] for color adjustments
        img = img / 255.0
        
        # 5. Apply saturation (in HSV space)
        saturation = cam_config.get('saturation', 1.0)
        if saturation != 1.0 and len(img.shape) == 3 and img.shape[2] == 3:
            # Convert RGB to HSV
            img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 1)
            img_hsv_uint8 = np.clip(img_hsv * 255, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img_hsv_uint8, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # 6. Apply gamma correction
        gamma = cam_config.get('gamma', 1.0)
        if gamma != 1.0:
            img = np.clip(img, 0, 1)  # Ensure valid range for gamma
            img = np.power(img, gamma)
        
        # 7. Apply RGB multipliers
        r_mul = cam_config.get('r_mul', 1.0)
        g_mul = cam_config.get('g_mul', 1.0)
        b_mul = cam_config.get('b_mul', 1.0)
        
        if len(img.shape) == 3 and img.shape[2] == 3 and (r_mul != 1.0 or g_mul != 1.0 or b_mul != 1.0):
            img[:, :, 0] = np.clip(img[:, :, 0] * r_mul, 0, 1)
            img[:, :, 1] = np.clip(img[:, :, 1] * g_mul, 0, 1)
            img[:, :, 2] = np.clip(img[:, :, 2] * b_mul, 0, 1)
        
        # Convert back to uint8 with proper clipping
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img

    def random_shift(self, img, shift_x=0, shift_y=0):
        max_shift_x = int(self.img_hw[1] * 0.2)
        max_shift_y = int(self.img_hw[0] * 0.2)

        if shift_x == 0 and shift_y == 0:
            shift_x = np.random.randint(-max_shift_x, max_shift_x)
            shift_y = np.random.randint(-max_shift_y, max_shift_y)

        img = T.functional.affine(img, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
        return img, shift_x, shift_y

    def __call__(self, sample, episode_path=None, phase_config=None):
        # Load wrist augmentation config for this episode
        aug_config = None
        if episode_path is not None:
            aug_config = self.load_episode_wrist_rotation(episode_path, phase_config=phase_config)
        
        # Convert tensors to numpy for Albumentations
        # Note: tensors are in [0, 1] range, convert to [0, 255] uint8 for processing
        sample_np = {}
        for k, v in sample.items():
            img_np = v.permute(1, 2, 0).cpu().numpy()
            # Convert from [0, 1] float to [0, 255] uint8
            if img_np.dtype in [np.float32, np.float64] and img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            elif img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            sample_np[k] = img_np
        
        # Apply wrist camera calibration-based augmentations BEFORE other augmentations
        # This includes rotation, brightness, contrast, saturation, gamma, and RGB multipliers
        if aug_config is not None:
            for key in ['left_wrist', 'right_wrist']:
                if key in sample_np:
                    sample_np[key] = self.apply_wrist_augmentation(sample_np[key], aug_config, key)

        # Apply spatial transforms (crop, resize, rotate) consistently
        if 'left' in sample_np and 'mask' in sample_np:
            aug = self.spatial_transforms(image=sample_np['left'], mask=sample_np['mask'])
            sample_np['left'] = aug['image']
            sample_np['mask'] = aug['mask']
        else:
            sample_np['left'] = self.spatial_transforms(image=sample_np['left'])['image']

        # Apply pixel dropout consistently
        if 'left' in sample_np and 'mask' in sample_np:
            aug = self.pixel_dropout(image=sample_np['left'], mask=sample_np['mask'])
            sample_np['left'] = aug['image']
            sample_np['mask'] = aug['mask']
        else:
            sample_np['left'] = self.pixel_dropout(image=sample_np['left'])['image']
            if 'left_wrist' in sample_np:
                sample_np['left_wrist'] = self.pixel_dropout(image=sample_np['left_wrist'])['image']
            if 'right_wrist' in sample_np:
                sample_np['right_wrist'] = self.pixel_dropout(image=sample_np['right_wrist'])['image']

        # Convert back to torch tensors
        processed = {}
        shift_x = shift_y = 0
        for key, img_np in sample_np.items():
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)

            # Apply color jitter only to RGB images
            if key in ['left', 'right', 'img_l_hist', 'img_lw', 'img_rw'] and img_t.shape[0] == 3:
                img_t = self.color_jitter(img_t)

            # Apply shift (shared between 'left' and 'mask')
            if key == 'left' or (self.stereo and key == 'right'):
                img_t, shift_x, shift_y = self.random_shift(img_t, shift_x=shift_x, shift_y=shift_y)
            elif key == 'mask':
                img_t, _, _ = self.random_shift(img_t, shift_x=shift_x, shift_y=shift_y)

            processed[key] = img_t

        # Optional sketch/history masking
        if 'img_l_hist' in processed and np.random.rand() < self.mask_sketch_prob:
            processed['img_l_hist'] = torch.zeros_like(processed['img_l_hist'])

        # Random full masking
        if np.random.rand() < self.mask_prob:
            mask_choice = np.random.choice(list(processed.keys()))
            processed[mask_choice] = torch.zeros_like(processed[mask_choice])

        return processed