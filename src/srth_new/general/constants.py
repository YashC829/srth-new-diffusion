import importlib.util
import json
import os
from pathlib import Path

THIRD_PERSON_CAM_DIR_NAME = "left_img_dir"
THIRD_PERSON_CAM_IMG_SUFFIX = "left"
PSM2_WRIST_CAM_DIR_NAME = "endo_psm2"
PSM2_WRIST_CAM_IMG_SUFFIX = "psm2"
PSM1_WRIST_CAM_DIR_NAME = "endo_psm1"
PSM1_WRIST_CAM_IMG_SUFFIX = "psm1"

LOW_LEVEL_DATASET_CAMERA_NAMES = ["left", "left_wrist", "right_wrist"]
LOW_LEVEL_DATASET_CAMERA_SUFFIXES = ["_left.jpg", "_psm2.jpg", "_psm1.jpg"]

CAMERA_NAMES = [THIRD_PERSON_CAM_DIR_NAME, PSM1_WRIST_CAM_DIR_NAME, PSM2_WRIST_CAM_DIR_NAME]
IMG_RESIZE_SIZE = (224, 224)

TISSUE_FOLDER_NAME = lambda i: f"tissue_{i}"

# dataset statistics caching
spec = importlib.util.find_spec("srth_new")
PACKAGE_ROOT = Path(spec.origin).resolve().parent.parent.parent
DATASET_STATS_CACHE_DIR = PACKAGE_ROOT.joinpath(".dataset_stats_cache")
DATASET_STATS_CACHE_FILE = DATASET_STATS_CACHE_DIR.joinpath("dataset_stats.json")
os.makedirs(DATASET_STATS_CACHE_DIR, exist_ok=True)
if not os.path.exists(DATASET_STATS_CACHE_FILE):
    with open(DATASET_STATS_CACHE_FILE, "w") as file:
        json.dump({}, file)
