import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import DictConfig

from srth_new.general import constants

def get_sorted_phases(tissue_dir: Path) -> List[str]:
    phases = [file_name for file_name in os.listdir(tissue_dir)]
    phases_ordered = sorted(phases, key=lambda x: int(x.split('_')[0]))
    return phases_ordered

def get_valid_ep_start_end_indices(
        ep_dir_path, # path to the episode directory
    ):
    # Load the start and end indices for the current demo as the valid range of the demo
    num_ep_frames = len(os.listdir(Path(ep_dir_path).joinpath(constants.LEFT_ENDOSCOPE_CAM_DIR_NAME)))
    start, end = 0, num_ep_frames - 1
    demo_num_frames_valid = end - start + 1

    return start, end, demo_num_frames_valid

def validate_selected_phases(phases: DictConfig) -> None:
    """The user will input a DictConfig object of phases that it wishes to use
    from the dataset. This function will take in that DictConfig of phases and
    validate it against the constant list of valid phases to ensure that a typo
    did not occur."""
    for high_level_phase, low_level_phase_list in phases.items():
        if high_level_phase not in constants.PHASES.keys():
            raise Exception(
                f"High level phase {high_level_phase} not found in list of valid high level phases: "
                f"{', '.join(list(constants.PHASES.keys()))}"
            )
        valid_low_level_phase_set = constants.PHASES[str(high_level_phase)]
        for low_level_phase in low_level_phase_list:
            if low_level_phase not in valid_low_level_phase_set:
                raise Exception(
                    f"Low level phase {low_level_phase} not found in list of valid low level phases: "
                    f"{', '.join(list(valid_low_level_phase_set))}"
                )

def get_episode_directories_by_tissue_id_and_phase(
        dataset_dir: str, tissue_ids: List[int], phases: DictConfig
    ) -> Tuple[List[str], Dict[str, int]]:
    samples = {}
    episode_dirs = list()

    # keep track of the number of episodes per low level phase
    num_episodes_info = dict()
    for id in tissue_ids:
        samples[id] = {}
        tissue_dir = os.path.join(dataset_dir, f"Tissue#{id}")
        annotator_name_dirs =[os.path.join(tissue_dir, x) for x in os.listdir(tissue_dir)]

        # Check if tissue directory exists
        if not os.path.exists(tissue_dir):
            raise Exception(f"Tissue directory does not exist: {tissue_dir}")
        
        for high_level_phase, low_level_phase_list in phases.items():
            for annotator_name_dir in annotator_name_dirs:
                high_level_phase_dir = os.path.join(annotator_name_dir, str(high_level_phase))
                # skip high level phase if no directory found in dataset. this just
                # means that high level phase wasn't collected for that tissue
                if not os.path.exists(high_level_phase_dir):
                    continue
                for low_level_phase in low_level_phase_list:
                    low_level_phase_dir = os.path.join(high_level_phase_dir, str(low_level_phase))
                    # skip low level phase if no directory found in dataset. this just
                    # means that low level phase wasn't collected for that tissue
                    if not os.path.exists(low_level_phase_dir):
                        continue
                    
                    # add to list of episode directories
                    temp_episode_dirs = [os.path.join(low_level_phase_dir, x) for x in os.listdir(low_level_phase_dir)]
                    episode_dirs.extend(temp_episode_dirs)

                    phase_desc = f"{high_level_phase}-{low_level_phase}"
                    if phase_desc not in num_episodes_info:
                        num_episodes_info[phase_desc] = 0
                    num_episodes_info[phase_desc] += len(temp_episode_dirs)

    return episode_dirs, num_episodes_info

def get_all_episode_directories(dataset_dir: str) -> Tuple[List[str], Dict[str, int]]:
    episode_dirs: List[str] = []
    num_episodes_info: Dict[str, int] = {}

    if not os.path.exists(dataset_dir):
        raise Exception(f"Dataset directory does not exist: {dataset_dir}")

    for tissue_name in os.listdir(dataset_dir):
        tissue_dir = os.path.join(dataset_dir, tissue_name)

        if not os.path.isdir(tissue_dir) or not tissue_name.startswith("Tissue#"):
            continue

        for annotator_name in os.listdir(tissue_dir):
            annotator_dir = os.path.join(tissue_dir, annotator_name)

            if not os.path.isdir(annotator_dir):
                continue

            for high_level_phase in os.listdir(annotator_dir):
                high_level_phase_dir = os.path.join(annotator_dir, high_level_phase)

                if not os.path.isdir(high_level_phase_dir):
                    continue

                for low_level_phase in os.listdir(high_level_phase_dir):
                    low_level_phase_dir = os.path.join(
                        high_level_phase_dir,
                        low_level_phase,
                    )

                    if not os.path.isdir(low_level_phase_dir):
                        continue

                    temp_episode_dirs = [
                        os.path.join(low_level_phase_dir, episode_name)
                        for episode_name in os.listdir(low_level_phase_dir)
                        if os.path.isdir(
                            os.path.join(low_level_phase_dir, episode_name)
                        )
                    ]

                    episode_dirs.extend(temp_episode_dirs)

                    phase_desc = f"{high_level_phase}-{low_level_phase}"
                    num_episodes_info[phase_desc] = (
                        num_episodes_info.get(phase_desc, 0)
                        + len(temp_episode_dirs)
                    )

    # remove phases that do not contain any episodes for brevity
    keys_to_delete = list()
    for phase, val in num_episodes_info.items():
        if val == 0:
            keys_to_delete.append(phase)
    for phase in keys_to_delete:
        del num_episodes_info[phase]

    return episode_dirs, num_episodes_info

def clean_low_level_phase_from_cfg(phase):
    return " ".join(phase.split("_")[1:])

def clean_high_level_phase_from_cfg(phase):
    return phase.replace("_", " ")

def get_low_level_phase_from_csv_path(csv_path: str) -> str:
    return " ".join(Path(csv_path).parts[-3].split("_")[1:])

def get_low_level_phase_from_ep_dir(ep_dir: str) -> str:
    return " ".join(Path(ep_dir).parts[-2].split("_")[1:])

def get_high_level_phase_from_ep_dir(ep_dir: str) -> str:
    return Path(ep_dir).parts[-3]

def get_collector_from_ep_dir(ep_dir: str) -> str:
    return Path(ep_dir).parts[-4]

def get_tissue_name_from_ep_dir(ep_dir: str) -> str:
    return Path(ep_dir).parts[-5]