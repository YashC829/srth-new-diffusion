from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from numpy.typing import NDArray
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from tqdm import tqdm

log = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    dataset_dir: str
    tissue_sample_ids_train: List[int]

def convert_6d_rot_to_quat(rots: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert a batch of 6D rotation representations to quaternions.

    Args:
        rots: NumPy array of shape ``(N, 6)``. Each row stores the first
            two columns of a rotation matrix, flattened as
            ``[c1_x, c1_y, c1_z, c2_x, c2_y, c2_z]``.

    Returns:
        NumPy array of shape ``(N, 4)`` with quaternions in ``xyzw``
        format.
    """
    c1 = rots[:, 0:3]
    c2 = rots[:, 3:6]
    c1 = normalize(c1, axis=1)
    dot_product = np.sum(c1 * c2, axis=1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product * c1, axis=1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3))
    rots = R.from_matrix(r_mat)
    return rots.as_quat()

def convert_actions_to_SE3_then_final_actions(
    dts: NDArray[np.float64],
    dquats: NDArray[np.float64],
    qpos_psm: NDArray[np.float64],
    jaw_angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compose delta task-space actions with the current pose.

    Args:
        dts: NumPy array of shape ``(N, 3)`` containing Cartesian position
            deltas ``[dx, dy, dz]``.
        dquats: NumPy array of shape ``(N, 4)`` containing delta
            quaternions in ``xyzw`` format.
        qpos_psm: NumPy array of shape ``(>=7,)`` for the current arm pose.
            Entries ``[0:3]`` are position and ``[3:7]`` is the current
            orientation quaternion in ``xyzw`` format.
        jaw_angles: NumPy array of shape ``(N,)`` or ``(N, 1)`` containing
            jaw commands in radians.

    Returns:
        NumPy array of shape ``(N, 8)`` with absolute task-space actions
        ordered as ``[x, y, z, qx, qy, qz, qw, jaw]``.
    """
    dquats = batch_rotations.batch_quaternion_wxyz_from_xyzw(dquats)
    qpos_psm[3:7] = rotations.quaternion_wxyz_from_xyzw(qpos_psm[3:7])
    dts_dquats = np.concatenate((dts, dquats), axis=1)
    g_qpos = transformations.transform_from_pq(qpos_psm[0:7])
    g_actions = trajectories.transforms_from_pqs(dts_dquats)
    g_poses = trajectories.concat_one_to_many(g_qpos, g_actions)
    output = np.zeros((dquats.shape[0], 8))
    output[:, 0:3] = g_poses[:, 0:3, 3]
    tmp = batch_rotations.quaternions_from_matrices(g_poses[:, 0:3, 0:3])
    output[:, 3:7] = batch_rotations.batch_quaternion_xyzw_from_wxyz(tmp)
    output[:, 7] = np.clip(jaw_angles, -0.698, 1.4)
    return output

def compute_diff_actions_relative_endoscope(
    qpos: NDArray[np.float64],
    action: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert absolute actions into deltas relative to the endoscope frame.

    Args:
        qpos: NumPy array of shape ``(9,)`` describing the current pose as
            ``[x, y, z, qx, qy, qz, qw, ..., jaw]``. The last entry is the
            jaw angle; the intermediate non-pose slot is kept as-is by the
            existing layout.
        action: NumPy array of shape ``(N, 9)`` containing absolute action
            targets in the same layout as ``qpos``.

    Returns:
        NumPy array of shape ``(N, 10)``. Translation is stored as a delta,
        orientation is converted to a 6D rotation representation in slots
        ``[3:9]``, and the final slot stores the absolute jaw angle.
    """
    # find diff first and then fill-in the quaternion differences properly
    diff = action - qpos
    quat_actions = action[:, 3:7]

    r_actions = R.from_quat(quat_actions)
    diff_rs = r_actions 
    # extract their first two columns
    diff_6d = diff_rs.as_matrix()[:,:,:2]
    diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
    diff_expand = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
    diff_expand[:diff.shape[0], 0:diff.shape[1]] = diff 
    diff = diff_expand

    diff[:, 3:9] = diff_6d
    diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
    return diff

def unnormalize_action_positions_only_min_max(
    naction: NDArray[np.float64],
    min: float,
    max: float
) -> NDArray[np.float64]:
    """
    Undo min-max normalization for action tensors with raw 6D rotation blocks.

    Args:
        naction: NumPy array of shape ``(N, D)`` containing normalized
            actions in the ``[-1, 1]`` range. Position-related entries are
            min-max normalized, while rotation slices ``[3:9]`` and
            ``[13:19]`` are already in raw 6D form.
        min: Scalar or per-dimension minimum used during min-max
            normalization.
        max: Scalar or per-dimension maximum used during min-max
            normalization.

    Returns:
        NumPy array of shape ``(N, D)`` with min-max normalized dimensions
        restored to the original scale.
    """
    action = None
    action = (naction + 1) / 2 * (max - min) + min
    action[:, 3:9] = naction[:, 3:9]
    action[:, 13:19] = naction[:, 13:19]
    return action

def unnormalize_positions_only_std(
    diffs: NDArray[np.float64],
    mean: float,
    std: float
) -> NDArray[np.float64]:
    """
    Undo standardization for action tensors with raw 6D rotation blocks.

    Args:
        diffs: NumPy array of shape ``(N, D)``. Position-related entries are
            assumed to be z-scored, while rotation slices ``[3:9]`` and
            ``[13:19]`` are already in raw 6D form.

    Returns:
        NumPy array of shape ``(N, D)`` with the standardized dimensions
        restored to the original scale.
    """
    unnormalized = diffs * std + mean
    unnormalized[:, 3:9] = diffs[:, 3:9]
    unnormalized[:, 13:19] = diffs[:, 13:19]
    return unnormalized

def convert_delta_6d_to_taskspace_quat(
    all_actions: NDArray[np.float64],
    all_actions_converted: NDArray[np.float64],
    qpos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert 6D delta rotations into absolute task-space quaternions.

    Args:
        all_actions: NumPy array of shape ``(N, >=9)`` where columns
            ``[3:9]`` store a 6D rotation representation.
        all_actions_converted: NumPy array of shape ``(N, M)`` that will be
            updated in-place. Quaternion outputs are written to ``[3:7]``.
        qpos: NumPy array of shape ``(>=7,)`` containing the current pose,
            with the current orientation quaternion in ``qpos[3:7]`` using
            ``xyzw`` format.

    Returns:
        The modified ``all_actions_converted`` array with absolute
        quaternions in ``xyzw`` format.
    """
    # Gram-schmidt
    c1 = all_actions[:, 3:6] # t x 3
    c2 = all_actions[:, 6:9] # t x 3 
    c1 = normalize(c1, axis = 1) # t x 3
    dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product*c1, axis = 1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
    # transform delta rot into task space
    rots = R.from_matrix(r_mat)
    rot_init = R.from_quat(qpos[3:7])
    rots = (rot_init * rots).as_quat()
    all_actions_converted[:, 3:7] = rots
    return all_actions_converted


def convert_delta_6d_to_taskspace_quat_relative_endo(
    all_actions: NDArray[np.float64],
    all_actions_converted: NDArray[np.float64],
    qpos: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert 6D delta rotations into quaternions without applying ``qpos``.

    Args:
        all_actions: NumPy array of shape ``(N, >=9)`` where columns
            ``[3:9]`` store a 6D rotation representation.
        all_actions_converted: NumPy array of shape ``(N, M)`` that will be
            updated in-place. Quaternion outputs are written to ``[3:7]``.
        qpos: Current pose array. It is accepted for interface consistency
            but is not used in this relative-endoscope variant.

    Returns:
        The modified ``all_actions_converted`` array with quaternions in
        ``xyzw`` format written to ``[3:7]``.
    """
    # Gram-schmidt
    c1 = all_actions[:, 3:6] # t x 3
    c2 = all_actions[:, 6:9] # t x 3 
    c1 = normalize(c1, axis = 1) # t x 3
    dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product*c1, axis = 1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
    # transform delta rot into task space
    rots = R.from_matrix(r_mat).as_quat()
    # rot_init = R.from_quat(qpos[3:7])
    # rots = (rot_init * rots).as_quat()
    all_actions_converted[:, 3:7] = rots
    return all_actions_converted

def average_quaternions(
    quaternions: NDArray[np.float64] | torch.Tensor,
    weights: NDArray[np.float64] | torch.Tensor,
) -> NDArray[np.float64]:
    """
    Average a set of quaternions using weighted averaging.
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()  # Move to CPU and convert to NumPy
        quaternions = quaternions.cpu().numpy()
    # Normalize weights to sum to 1
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    # Quaternion averaging: weighted mean in quaternion space
    avg_quat = np.zeros((4,))
    for i, quat in enumerate(quaternions):
        avg_quat += weights[i] * quat

    # Normalize the resulting quaternion
    avg_quat /= np.linalg.norm(avg_quat)
    return avg_quat


def compute_relative_actions_in_SE3(
    qpos: NDArray[np.float64],
    action: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert absolute task-space actions into SE(3)-relative deltas.

    Args:
        qpos: NumPy array of shape ``(8,)`` describing the current pose as
            ``[x, y, z, qx, qy, qz, qw, jaw]``.
        action: NumPy array of shape ``(N, 8)`` containing absolute target
            actions in the same layout as ``qpos``.

    Returns:
        NumPy array of shape ``(N, 10)``. Translation is expressed relative to
        ``qpos`` in SE(3), orientation is stored as a 6D rotation
        representation in slots ``[3:9]``, and the final slot stores the
        absolute jaw angle.
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

def compute_quat_diff(
    quat1: NDArray[np.float64],
    quat2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the relative quaternion from ``quat1`` to ``quat2``.

    Args:
        quat1: NumPy array of shape ``(4,)`` containing a reference
            quaternion in ``xyzw`` format.
        quat2: NumPy array of shape ``(N, 4)`` or ``(4,)`` containing target
            quaternions in ``xyzw`` format.

    Returns:
        NumPy array containing the quaternion difference in ``xyzw`` format.
        The output shape follows the shape of ``quat2``.
    """
    r1 = R.from_quat(quat1)  # single element
    r2 = R.from_quat(quat2)  # many rows of elements
    diff = r1.inv() * r2
    diff = diff.as_quat()
    return diff

def computer_diff_actions(
    qpos: NDArray[np.float64],
    action: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert absolute actions into pose-relative deltas using quaternion math.

    Args:
        qpos: NumPy array of shape ``(8,)`` describing the current pose as
            ``[x, y, z, qx, qy, qz, qw, jaw]``.
        action: NumPy array of shape ``(N, 8)`` containing absolute target
            actions in the same layout as ``qpos``.

    Returns:
        NumPy array of shape ``(N, 10)``. Translation is stored as a delta,
        orientation is converted to a 6D rotation representation in slots
        ``[3:9]``, and the final slot stores the absolute jaw angle.
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

def compute_diff_actions_wrt_camera(
    qpos: NDArray[np.float64],
    action: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert absolute actions into camera-frame deltas.

    Args:
        qpos: NumPy array of shape ``(9,)`` describing the current pose as
            ``[x, y, z, qx, qy, qz, qw, ..., jaw]``.
        action: NumPy array of shape ``(N, 9)`` containing absolute action
            targets in the same layout as ``qpos``.

    Returns:
        NumPy array of shape ``(N, 10)``. Translation is stored as a delta,
        orientation is converted directly from the absolute quaternion into a
        6D rotation representation in slots ``[3:9]``, and the final slot
        stores the absolute jaw angle.
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

def compute_diffs(
    ids: List[int],
    data_dir: Path | str,
    chunk_size: int = 100,
    phantoms: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Compute dataset-wide action normalization statistics from demonstration CSVs.

    Args:
        ids: Tissue or phantom identifiers to scan under ``data_dir``.
        data_dir: Root directory containing ``tissue_*`` or ``phantom_*``
            subdirectories.
        chunk_size: Number of future setpoint rows to include for each current
            pose when building action-difference chunks.
        phantoms: If ``True``, read from ``phantom_*`` directories; otherwise
            read from ``tissue_*`` directories.

    Returns:
        A 4-tuple ``(mean, std, min, max)`` where each element is a NumPy array
        of per-dimension statistics computed over the stacked relative-action
        dataset.
    """
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
def generate_task_config(
    data_dir: Path | str,
    tissue_ids: List[int],
) -> None:
    """
    Generate and save normalization statistics for a set of tissue datasets.

    Args:
        data_dir: Root directory containing the demonstration folders.
        tissue_ids: Tissue identifiers to include when computing statistics.

    Returns:
        ``None``. The computed statistics are logged and written to
        ``./std_mean_invivo.txt``.
    """
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
