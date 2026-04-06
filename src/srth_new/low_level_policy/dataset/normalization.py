from typing import List

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def compute_relative_actions_in_SE3(qpos, action):
    """
    Note: this is the proper implementation
    qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
    action: set point on the dvrk (action_horizon x 8)

    returns: relative position and rotation w.r.t qpos
    """

    diff = np.zeros((action.shape[0], 10))  # TODO: hard-coded dim (10) for a single arm

    # convert current pose to SE(3)
    qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
    qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
    g_qpos = transformations.transform_from_pq(qpos_py3d)  # no jaw angle!

    # convert actions to SE(3)
    action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7])
    action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis=1)
    g_action = trajectories.transforms_from_pqs(action_py3d)

    # invert current pose
    g_qpos_inv = transformations.invert_transform(g_qpos)
    diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

    # construct 6d rot
    diff_6d = diff_SE3[:, 0:3, :2]
    diff_6d = diff_6d.transpose(0, 2, 1).reshape(-1, 6)  # first column then second column

    # fill in translation elements
    diff[:, 0:3] = diff_SE3[:, 0:3, 3]  # replace the translations with the last column first three rows of SE3
    # fill in 6d rot
    diff[:, 3:9] = diff_6d
    # fill in jaw angle (note: jaw angle is absolute, not relative)
    diff[:, 9] = action[:, 7]
    return diff


def compute_quat_diff(quat1, quat2):
    r1 = R.from_quat(quat1)  # single element
    r2 = R.from_quat(quat2)  # many rows of elements
    diff = r1.inv() * r2
    diff = diff.as_quat()
    return diff


def computer_diff_actions(qpos, action):
    """
    qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
    action: set point on the dvrk (action_horizon x 8)

    returns: relative position and rotation w.r.t qpos
    """

    # find diff first and then fill-in the quaternion differences properly
    diff = action - qpos

    quat_init = qpos[3:7]
    quat_actions = action[:, 3:7]

    # convert quaternions to rotation matrices
    r_init = R.from_quat(quat_init)
    r_actions = R.from_quat(quat_actions)
    # find their diff
    diff_rs = r_init.inv() * r_actions
    # extract their first two columns
    diff_6d = diff_rs.as_matrix()[:, :, :2]
    diff_6d = diff_6d.transpose(0, 2, 1).reshape(-1, 6)  # first column then second column

    diff_exp = np.zeros((diff.shape[0], 10))  # TODO: hard-coded dim (10) for a single arm
    diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff
    diff = diff_exp

    diff[:, 3:9] = diff_6d
    diff[:, 9] = action[:, -1]  # fill in the jaw angle (note: jaw angle is not relative)
    return diff


def compute_diff_actions_wrt_camera(qpos, action):
    """
    qpos: current position [9]
    action: actions commanded by the user [n_actions x 9]
    returns: relative actions w.r.t qpos
    """
    # find diff first and then fill-in the quaternion differences properly
    diff = action - qpos
    quat_actions = action[:, 3:7]

    # convert quaternions to rotation matrices
    r_actions = R.from_quat(quat_actions)

    # extract their first two columns
    diff_6d = r_actions.as_matrix()[:, :, :2]
    diff_6d = diff_6d.transpose(0, 2, 1).reshape(-1, 6)  # first column then second column

    diff_exp = np.zeros((diff.shape[0], 10))  # TODO: hard-coded dim (10) for a single arm
    diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff
    diff = diff_exp

    diff[:, 3:9] = diff_6d
    diff[:, 9] = action[:, -1]  # fill in the jaw angle (note: jaw angle is not relative)
    return diff


def compute_diffs(ids, data_dir, chunk_size=100, phantoms=False):
    cp_psm1 = [
        "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
        "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
        "psm1_jaw",
    ]

    sp_psm1 = [
        "psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
        "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
        "psm1_jaw_sp",
    ]

    cp_psm2 = [
        "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
        "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
        "psm2_jaw",
    ]

    sp_psm2 = [
        "psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
        "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
        "psm2_jaw_sp",
    ]

    t = 0
    samples = {}

    for id in ids:
        samples[id] = {}
        if phantoms:
            root = os.path.join(data_dir, f"phantom_{id}")
        else:
            root = os.path.join(data_dir, f"tissue_{id}")

        # Check if root directory exists
        if not os.path.exists(root):
            log.info("Warning: Directory %s does not exist, skipping...", root)
            continue

        dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        dirlist = natsorted(dirlist)

        total_demo_num = 0
        for dir in dirlist:
            phase = os.path.join(root, dir)
            samples[id][dir] = []
            # Filter only directories, not files
            for item in os.listdir(phase):
                item_path = os.path.join(phase, item)
                if os.path.isdir(item_path):
                    samples[id][dir].append(item)
            total_demo_num += len(samples[id][dir])
        t += total_demo_num
        log.debug("Tissue %s - Total Num Episodes = %s", id, total_demo_num)

    log.debug("All Tissues: Total Num Episodes = %s", t)

    diffs = []

    total_rows = 0
    for id in ids:
        if phantoms:
            root = os.path.join(data_dir, f"phantom_{id}")
        else:
            root = os.path.join(data_dir, f"tissue_{id}")

        if not os.path.exists(root):
            continue

        for phase in samples.get(id, {}).keys():
            sample = samples[id][phase]
            for s in sample:
                if s == "Corrections":
                    sample_dir = os.path.join(root, phase, s)
                    new_sample = os.listdir(sample_dir)
                    for ss in new_sample:
                        sample_dir = os.path.join(sample_dir, ss)
                        break
                    pth = os.path.join(sample_dir, "ee_csv.csv")
                else:
                    pth = os.path.join(root, phase, s, "ee_csv.csv")

                if not os.path.exists(pth) or not os.path.isfile(pth):
                    continue

                try:
                    csv = pd.read_csv(pth)
                    total_rows += len(csv)
                except Exception:
                    continue

    pbar = tqdm(total=total_rows, desc="Computing action statistics", unit="row")

    try:
        for id in ids:
            # log.info("id: %s", id)
            if phantoms:
                root = os.path.join(data_dir, f"phantom_{id}")
            else:
                root = os.path.join(data_dir, f"tissue_{id}")

            # Check if root directory exists
            if not os.path.exists(root):
                log.info("Warning: Directory %s does not exist, skipping...", root)
                continue

            dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
            dirlist = natsorted(dirlist)
            for phase in samples[id].keys():
                sample = samples[id][phase]
                for s in sample:
                    if s == "Corrections":
                        sample_dir = os.path.join(root, phase, s)
                        new_sample = os.listdir(sample_dir)
                        for ss in new_sample:
                            sample_dir = os.path.join(sample_dir, ss)
                            break
                        pth = os.path.join(sample_dir, "ee_csv.csv")
                    else:
                        # TODO: change ee name
                        # pth = os.path.join(root, phase, s, "ee_estimate.csv")
                        pth = os.path.join(root, phase, s, "ee_csv.csv")

                    # Check if the CSV file exists before trying to read it
                    if not os.path.exists(pth):
                        log.info("Warning: CSV file %s does not exist, skipping...", pth)
                        continue

                    # Check if path is actually a file (not a directory)
                    if not os.path.isfile(pth):
                        log.info("Warning: %s is not a file, skipping...", pth)
                        continue

                    try:
                        csv = pd.read_csv(pth)
                    except Exception as e:
                        log.info("Warning: Failed to read CSV file %s: %s, skipping...", pth, e)
                        continue

                    for jj in range(len(csv)):
                        first_el_psm1 = csv[cp_psm1].iloc[jj, :].to_numpy()
                        chunk_el_psm1 = csv[sp_psm1].iloc[jj:jj + chunk_size, :].to_numpy()
                        # diff_psm1 = computer_diff_actions(first_el_psm1, chunk_el_psm1)
                        diff_psm1 = compute_relative_actions_in_SE3(first_el_psm1, chunk_el_psm1)

                        first_el_psm2 = csv[cp_psm2].iloc[jj, :].to_numpy()
                        chunk_el_psm2 = csv[sp_psm2].iloc[jj:jj + chunk_size, :].to_numpy()
                        diff_psm2 = computer_diff_actions(first_el_psm2, chunk_el_psm2)

                        diff_psm2 = compute_relative_actions_in_SE3(first_el_psm2, chunk_el_psm2)

                        diff_stacked = np.column_stack((diff_psm1, diff_psm2))
                        diffs.append(diff_stacked)
                        pbar.update(1)

            log.debug("Current diff chunks accumulated: %s", len(diffs))
    finally:
        pbar.close()

    diffs_np = np.concatenate(diffs, axis=0)
    mean = diffs_np.mean(axis=0)
    std = diffs_np.std(axis=0).clip(1e-2, 10)
    min = diffs_np.min(axis=0)
    max = diffs_np.max(axis=0)

    return mean, std, min, max


# Define the main function to generate the task configuration file
def generate_task_config(data_dir: Path | str, tissue_ids: List[int]):
    mean, std, min, max = compute_diffs(tissue_ids, data_dir)

    std_str = ", ".join(map(str, std))
    mean_str = ", ".join(map(str, mean))
    min_str = ", ".join(map(str, min))
    max_str = ", ".join(map(str, max))

    log.info("mean: %s", mean_str)
    log.info("std: %s", std_str)
    log.info("min: %s", min_str)
    log.info("max: %s", max_str)

    # write the results into a txt file
    with open("./std_mean_invivo.txt", "w") as f:
        f.write(f"tissue ids: {tissue_ids}\n")
        f.write(f"mean: {mean_str}\n")
        f.write(f"std: {std_str}\n")
        f.write(f"min: {min_str}\n")
        f.write(f"max: {max_str}\n")


# from typing import List

# import os
# import numpy as np
# import pandas as pd
# from pathlib import Path

# from natsort import natsorted
# from scipy.spatial.transform import Rotation as R
# from pytransform3d import rotations, batch_rotations, transformations, trajectories

# import logging
# log = logging.getLogger(__name__)

# def compute_relative_actions_in_SE3(qpos, action):
#     """
#     Note: this is the proper implementation
#     qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
#     action: set point on the dvrk (action_horizon x 8)
    
#     returns: relative position and rotation w.r.t qpos
#     """
    
#     diff = np.zeros((action.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm

#     # convert current pose to SE(3)
#     qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
#     qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
#     g_qpos = transformations.transform_from_pq(qpos_py3d) # no jaw angle!

#     # convert actions to SE(3)
#     action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7]) 
#     action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis = 1)
#     g_action = trajectories.transforms_from_pqs(action_py3d)

#     # invert current pose
#     g_qpos_inv = transformations.invert_transform(g_qpos)
#     diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

#     # construct 6d rot
#     diff_6d = diff_SE3[:,0:3,:2]
#     diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
#     # fill in translation elements
#     diff[:, 0:3] = diff_SE3[:, 0:3, 3] # replace the translations with the last column first three rows of SE3
#     # fill in 6d rot
#     diff[:, 3:9] = diff_6d
#     # fill in jaw angle (note: jaw angle is absolute, not relative)
#     diff[:, 9] = action[:, 7]
#     return diff

# def compute_quat_diff(quat1, quat2):
#     r1 = R.from_quat(quat1) # single element
#     r2 = R.from_quat(quat2) # many rows of elements
#     diff = r1.inv()*r2
#     diff = diff.as_quat()
#     return diff

# def computer_diff_actions(qpos, action):

#     """
#     qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
#     action: set point on the dvrk (action_horizon x 8)
    
#     returns: relative position and rotation w.r.t qpos
#     """

#     # find diff first and then fill-in the quaternion differences properly
#     diff = action - qpos

#     quat_init = qpos[3:7]
#     quat_actions = action[:, 3:7]

#     # convert quaternions to rotation matrices
#     r_init = R.from_quat(quat_init)
#     r_actions = R.from_quat(quat_actions)
#     # find their diff
#     diff_rs = r_init.inv()*r_actions 
#     # extract their first two columns
#     diff_6d = diff_rs.as_matrix()[:,:,:2]
#     diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
#     diff_exp = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
#     diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff
#     diff = diff_exp

#     diff[:, 3:9] = diff_6d
#     diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
#     return diff

# def compute_diff_actions_wrt_camera(qpos, action):
#         """
#         qpos: current position [9]
#         action: actions commanded by the user [n_actions x 9]
#         returns: relative actions w.r.t qpos
#         """
#         # find diff first and then fill-in the quaternion differences properly
#         diff = action - qpos
#         quat_actions = action[:, 3:7]

#         # convert quaternions to rotation matrices
#         r_actions = R.from_quat(quat_actions)

#         # extract their first two columns
#         diff_6d = r_actions.as_matrix()[:,:,:2]
#         diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
#         diff_exp = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
#         diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff 
#         diff = diff_exp

#         diff[:, 3:9] = diff_6d
#         diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
#         return diff    

# def compute_diffs(ids, data_dir, chunk_size=100, phantoms=False):
#     cp_psm1 = [ "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
#             "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
#             "psm1_jaw"]

#     sp_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
#             "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
#             "psm1_jaw_sp"]

#     cp_psm2 = [ "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
#             "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
#             "psm2_jaw"]

#     sp_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
#             "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
#             "psm2_jaw_sp"]

#     t = 0
#     samples = {}

#     for id in ids:
#         samples[id] = {}
#         if phantoms:
#             root = os.path.join(data_dir, f"phantom_{id}")
#         else:
#             root = os.path.join(data_dir, f"tissue_{id}")
        
#         # Check if root directory exists
#         if not os.path.exists(root):
#             print(f"Warning: Directory {root} does not exist, skipping...")
#             continue
            
#         print(root)
#         dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
#         dirlist = natsorted(dirlist)

#         total_demo_num = 0
#         for dir in dirlist:
#             phase = os.path.join(root, dir)
#             samples[id][dir] = []
#             # Filter only directories, not files
#             for item in os.listdir(phase):
#                 item_path = os.path.join(phase, item)
#                 if os.path.isdir(item_path):
#                     samples[id][dir].append(item)
#             total_demo_num += len(samples[id][dir])
#         t += total_demo_num
#         print(id, ", total demo num =", total_demo_num)
#     print("total demo num =", t)
    
#     diffs = []

#     for id in ids:
#         print("id:", id)
#         if phantoms:
#             root = os.path.join(data_dir, f"phantom_{id}")
#         else:
#             root = os.path.join(data_dir, f"tissue_{id}")
        
#         # Check if root directory exists
#         if not os.path.exists(root):
#             print(f"Warning: Directory {root} does not exist, skipping...")
#             continue
            
#         dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
#         dirlist = natsorted(dirlist)
#         for phase in samples[id].keys():

#             sample = samples[id][phase]
#             for s in sample:
#                 if s == "Corrections":
#                     sample_dir = os.path.join(root, phase, s)
#                     new_sample = os.listdir(sample_dir)
#                     for ss in new_sample:
#                         sample_dir = os.path.join(sample_dir, ss)
#                         break
#                     pth = os.path.join(sample_dir, "ee_csv.csv")
#                 else:
#                     ## TODO: change ee name
#                     # pth = os.path.join(root, phase, s, "ee_estimate.csv")
#                     pth = os.path.join(root, phase, s, "ee_csv.csv")
                
#                 # Check if the CSV file exists before trying to read it
#                 if not os.path.exists(pth):
#                     print(f"Warning: CSV file {pth} does not exist, skipping...")
#                     continue
                
#                 # Check if path is actually a file (not a directory)
#                 if not os.path.isfile(pth):
#                     print(f"Warning: {pth} is not a file, skipping...")
#                     continue
                
#                 try:
#                     csv = pd.read_csv(pth)
#                 except Exception as e:
#                     print(f"Warning: Failed to read CSV file {pth}: {e}, skipping...")
#                     continue

#                 for jj in range(len(csv)):
                    
#                     first_el_psm1 = csv[cp_psm1].iloc[jj, :].to_numpy()
#                     chunk_el_psm1 = csv[sp_psm1].iloc[jj:jj+chunk_size, :].to_numpy()
#                     #diff_psm1 = computer_diff_actions(first_el_psm1, chunk_el_psm1)
#                     diff_psm1 = compute_relative_actions_in_SE3(first_el_psm1, chunk_el_psm1)

#                     first_el_psm2 = csv[cp_psm2].iloc[jj, :].to_numpy()
#                     chunk_el_psm2 = csv[sp_psm2].iloc[jj:jj+chunk_size, :].to_numpy()
#                     diff_psm2 = computer_diff_actions(first_el_psm2, chunk_el_psm2)

#                     diff_psm2 = compute_relative_actions_in_SE3(first_el_psm2, chunk_el_psm2)

#                     diff_stacked = np.column_stack((diff_psm1, diff_psm2))
#                     diffs.append(diff_stacked)

#         print(len(diffs))

#     diffs_np = np.concatenate(diffs, axis=0)
#     mean = diffs_np.mean(axis=0)
#     std = diffs_np.std(axis=0).clip(1e-2, 10)
#     min = diffs_np.min(axis = 0)
#     max = diffs_np.max(axis = 0)

#     return mean, std, min, max

# # Define the main function to generate the task configuration file
# def generate_task_config(data_dir: Path | str, tissue_ids: List[int]):
    
#     mean, std, min, max = compute_diffs(tissue_ids, data_dir)

#     std_str = ', '.join(map(str, std))
#     mean_str = ', '.join(map(str, mean))
#     min_str = ', '.join(map(str, min))
#     max_str = ', '.join(map(str, max))

#     print("mean:", mean_str)
#     print("std:", std_str)
#     print("min:", min_str)
#     print("max:", max_str)

#     # write the results into a txt file
#     with open("./std_mean_invivo.txt", "w") as f:
#         f.write(f"tissue ids: {ids}\n")
#         f.write(f"mean: {mean_str}\n")
#         f.write(f"std: {std_str}\n")
#         f.write(f"min: {min_str}\n")
#         f.write(f"max: {max_str}\n")
