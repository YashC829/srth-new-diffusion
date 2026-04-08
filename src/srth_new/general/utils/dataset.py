import os
from pathlib import Path
from typing import List

from srth_new.general import constants

def get_sorted_phases(tissue_dir: Path) -> List[str]:
    phases = [file_name for file_name in os.listdir(tissue_dir)]
    phases_ordered = sorted(phases, key=lambda x: int(x.split('_')[0]))
    return phases_ordered

def get_valid_ep_start_end_indices(
        ep_dir_path, # path to the episode directory
    ):
    # Load the start and end indices for the current demo as the valid range of the demo
    num_ep_frames = len(os.listdir(Path(ep_dir_path).joinpath(constants.THIRD_PERSON_CAM_DIR_NAME)))
    start, end = 0, num_ep_frames - 1
    demo_num_frames_valid = end - start + 1
    
    return start, end, demo_num_frames_valid