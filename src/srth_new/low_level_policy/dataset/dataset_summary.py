#!/usr/bin/env python3
import os
import argparse
from typing import Dict

# Importing from your existing logic
import getpass
import os

from srth_new.general import constants

VALID_COLLECTORS = ["Antony", "Jacob", "Grayson", "Jaxon", "Megan", "Brianna", "Latavya", "Athena", "Melie", "Philip"]

PHASES = [
    "unzipping",
    "calot_dissection",
    "clipping_and_cutting",
    "gallbladder_removal",
]

ACTIONS = {
    PHASES[0]: [
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
    PHASES[1]: [
        "1_grabbing_gallbladder_right",
        "1_grabbing_gallbladder_right_recovery",
        "2_move_camera_down",
        "3_forceps_approach",
        "3_forceps_approach_recovery",
        "4_forceps_open",
    ],
    PHASES[2]: [
        "1_grabbing_gallbladder",
        "1_grabbing_gallbladder_recovery",
        "2_clipping_first_clip_left_tube",
        "2_clipping_first_clip_left_tube_recovery",
        "3_going_back_first_clip_left_tube",
        "4_clipping_second_clip_left_tube",
        "4_clipping_second_clip_left_tube_recovery",
        "5_going_back_second_clip_left_tube",
        "6_clipping_third_clip_left_tube",
        "6_clipping_third_clip_left_tube_recovery",
        "7_going_back_third_clip_left_tube",
        "8_go_to_the_cutting_position_left_tube",
        "8_go_to_the_cutting_position_left_tube_recovery",
        "9_go_back_from_the_cut_left_tube",
        "10_clipping_first_clip_right_tube",
        "10_clipping_first_clip_right_tube_recovery",
        "11_going_back_first_clip_right_tube",
        "12_clipping_second_clip_right_tube",
        "12_clipping_second_clip_right_tube_recovery",
        "13_going_back_second_clip_right_tube",
        "14_clipping_third_clip_right_tube",
        "14_clipping_third_clip_right_tube_recovery",
        "15_going_back_third_clip_right_tube",
        "16_go_to_the_cutting_position_right_tube",
        "16_go_to_the_cutting_position_right_tube_recovery",
        "17_go_back_from_the_cut_right_tube",
    ],
    PHASES[3]: [
        "1_grabbing_gallbladder_bottom_up",
        "1_grabbing_gallbladder_bottom_up_recovery",
        "2_hook_tissue",
        "2_hook_tissue_recovery",
        "3_pull_and_burn",
        "4_pull_up",
        "5_zoom_out",
    ],
}

def _existing_tissue_numbers(base_dir: str) -> list[int]:
    if not os.path.isdir(base_dir):
        return []

    existing = []
    for entry in os.listdir(base_dir):
        if entry.startswith("Tissue#"):
            try:
                existing.append(int(entry[len("Tissue#") :]))
            except ValueError:
                pass
    return existing


def get_next_tissue_num_for_creator(base_dir: str) -> int:
    existing = _existing_tissue_numbers(base_dir)
    return max(existing) + 1 if existing else 1


def get_next_tissue_num_for_recorder(base_dir: str) -> int:
    existing = _existing_tissue_numbers(base_dir)
    return max(existing) if existing else 1

def count_episodes_and_frames(save_dir: str) -> tuple[int, int]:
    if not os.path.isdir(save_dir):
        return 0, 0

    episode_count = 0
    frame_count = 0
    
    with os.scandir(save_dir) as entries:
        for entry in entries:
            name = entry.name
            # Your existing naming convention check
            if (
                len(name) == 22
                and name[8] == "-"
                and name[15] == "-"
                and name[:8].isdigit()
                and name[9:15].isdigit()
                and name[16:].isdigit()
                and entry.is_dir(follow_symlinks=False)
            ):
                episode_count += 1
                
                # Path to the frames folder
                img_dir = os.path.join(entry.path, "left_img_dir")
                if os.path.isdir(img_dir):
                    # Efficiently count only .jpg files
                    with os.scandir(img_dir) as img_entries:
                        frame_count += sum(1 for img in img_entries if img.name.lower().endswith('.jpg'))
                        
    return episode_count, frame_count


def generate_summary(dataset_name: str):
    dataset_base_dir = str(constants.RAW_DATASET_ROOT)
    
    stats = {}
    grand_total_samples = 0
    total_frames = 0
    fps = 30

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"Source: {dataset_base_dir}")
    print(f"{'='*60}\n")

    if not os.path.exists(dataset_base_dir):
        print("Dataset directory not found.")
        return

    tissues = [d for d in os.listdir(dataset_base_dir) if d.startswith("Tissue#")]
    
    for tissue in tissues:
        tissue_path = os.path.join(dataset_base_dir, tissue)
        for collector in VALID_COLLECTORS:
            collector_path = os.path.join(tissue_path, collector)
            if not os.path.isdir(collector_path):
                continue
            
            for phase in PHASES:
                stats.setdefault(phase, {})
                for action in ACTIONS.get(phase, []):
                    stats[phase].setdefault(action, {})
                    
                    action_path = os.path.join(collector_path, phase, action)
                    
                    # Get both counts from the updated helper
                    count, frames = count_episodes_and_frames(action_path)
                    
                    if count > 0:
                        stats[phase][action][collector] = stats[phase][action].get(collector, 0) + count
                        grand_total_samples += count
                        total_frames += frames

    # --- Neat Printing ---
    for phase, actions in stats.items():
        print(f"PHASE: {phase.upper()}")
        print("-" * 40)
        
        phase_total = 0
        for action, collectors in actions.items():
            action_total = sum(collectors.values())
            phase_total += action_total
            print(f"  [{action_total:3}] {action}")
            for collector, count in collectors.items():
                print(f"        └─ {collector}: {count}")
        
        print(f"\n  Phase Total: {phase_total}")
        print("." * 40 + "\n")

    # Final Calculation
    total_seconds = total_frames / fps
    total_hours = total_seconds / 3600

    print(f"GRAND TOTAL SAMPLES: {grand_total_samples}")
    print(f"TOTAL DURATION     : {total_hours:.2f} hours")
    print(f"TOTAL FRAMES       : {total_frames} (@ {fps} fps)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Cholecystectomy dataset samples.")
    parser.add_argument("--dataset", type=str, default="Cholecystectomy", help="Dataset name")
    args = parser.parse_args()

    generate_summary(args.dataset)