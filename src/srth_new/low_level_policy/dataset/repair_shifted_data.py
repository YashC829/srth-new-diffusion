import csv
from pathlib import Path
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from glob import glob

from srth_new.general.utils import dataset


def repair_csv(input_path, output_path):
    """Realign 228-col CSVs by removing PSM3's 7th joint from the data rows."""
    with open(input_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(header) == 224 and rows and len(rows[0]) == 228:
        # These are the four extra data columns corresponding to missing header names.
        # Remove from highest index to lowest so earlier deletions do not shift later ones.
        remove_indices = [200, 193, 186, 179]

        fixed_rows = []
        for row in rows:
            if len(row) == 228:
                for idx in remove_indices:
                    if idx < len(row):
                        row.pop(idx)
            fixed_rows.append(row)

        rows = fixed_rows

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


dataset_dir = "/mnt/sda1/surpass_data/Cholecystectomy"
tissue_ids = list(range(1, len(glob(os.path.join(dataset_dir, "*")))+1))
tissue_ids = [24]
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

ep_dirs, num_ep_info = dataset.get_episode_directories_by_tissue_id_and_phase(
    dataset_dir, tissue_ids, phases
)

for ep_dir in tqdm(ep_dirs):
    original_csv_file = Path(ep_dir) / "ee_csv.csv"
    repaired_csv_temp_name = str(original_csv_file).replace(".csv", "_repaired.csv")

    repair_csv(str(original_csv_file), repaired_csv_temp_name)

    # move the corrupted original csv file to a new name
    os.rename(
        str(original_csv_file),
        str(original_csv_file).replace(".csv", "_corrupted_original.csv"),
    )

    # move the repaired csv file to the correct, original name
    os.rename(repaired_csv_temp_name, str(original_csv_file))



# import csv

# from srth_new.general.utils import dataset

# def repair_csv(input_path, output_path):
#     """Realign 228-col CSVs by extracting PSM3's 7th joint into new columns."""
#     with open(input_path) as f:
#         reader = csv.reader(f)
#         header = next(reader)
#         rows = list(reader)

#     if len(header) == 224 and rows and len(rows[0]) == 228:
#         # Insert 4 new header entries at the correct positions
#         # PSM3 section starts at index 173 (0-based) in the header
#         # Indices where the extra elements appear:
#         #   After psm3_js[5]       → index 179 → insert psm3_js[6]
#         #   After psm3_js_effort[5] → index 185+1=186 → insert psm3_js_effort[6]
#         #   After psm3_js_velocity[5] → index 191+2=193 → insert psm3_js_velocity[6]
#         #   After psm3_set_js[5]   → index 197+3=200 → insert psm3_set_js[6]
#         inserts = [
#             (179, "psm3_js[6]"),
#             (186, "psm3_js_effort[6]"),
#             (193, "psm3_js_velocity[6]"),
#             (200, "psm3_set_js[6]"),
#         ]
#         for offset, (pos, name) in enumerate(inserts):
#             header.insert(pos + offset, name)

#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         writer.writerows(rows)

# from omegaconf import OmegaConf
# dataset_dir = '/mnt/sda1/surpass_data/Cholecystectomy'
# tissue_ids = [15, 16, 17, 18]
# phases = OmegaConf.create(
#     {
#         "unzipping": [
#             "1_grabbing_gallbladder_right",
#             "1_grabbing_gallbladder_right_recovery",
#             "2_initial_incision",
#             "2_initial_incision_recovery",
#             "3_hook_to_local_home",
#             "4_hook_tissue",
#             "4_hook_tissue_recovery",
#             "5_cauterize_tissue_right",
#             "6_hook_to_global_home",
#             "7_grasper_to_home",
#             "8_grabbing_gallbladder_left",
#             "8_grabbing_gallbladder_left_recovery",
#             "9_returning_to_initial_incision",
#             "9_returning_to_initial_incision_recovery",
#             "10_cauterize_tissue_left",
#             "11_regrab",
#             "12_hook_to_global_home",
#             "13_grasper_to_home",
#         ],
#     }
# )
# ep_dirs, num_ep_info = dataset.get_episode_directories_by_tissue_id_and_phase(
#     dataset_dir, tissue_ids, phases
# )

# from pathlib import Path
# from tqdm import tqdm
# import os

# for ep_dir in tqdm(ep_dirs):
#     original_csv_file = Path(ep_dir) / "ee_csv.csv"
#     repaired_csv_temp_name = str(original_csv_file).replace(".csv", "_repaired.csv")
#     repair_csv(str(original_csv_file), repaired_csv_temp_name)
    
#     # move the corrupted original csv file to a new name
#     os.rename(str(original_csv_file), str(original_csv_file).replace(".csv", "_corrupted_original.csv"))
    
#     # move the repaired csv file to the correct, original name
#     os.rename(repaired_csv_temp_name, str(original_csv_file))
