import argparse
import json
import os
import re
import shutil
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from srth_new.general import constants
from srth_new.general.utils import dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create LeRobot dataset")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--no-hardlink", action="store_true")
    parser.add_argument("--disable-preplace", action="store_true")
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    return parser.parse_args()


def delete_hf_dataset(repo_id: str):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    path = os.path.join(hf_home, "lerobot", repo_id)

    if not os.path.exists(path):
        return

    response = ""
    while response not in ["yes", "no"]:
        response = input(
            f"Delete previously saved dataset: {path}? Type 'yes' or 'no': "
        )

    if response == "no":
        raise RuntimeError("User aborted conversion.")

    shutil.rmtree(path)


def print_ep_info(info):
    print("\n" + "-" * 100)
    print("Dataset Episode Information:")
    print(json.dumps(info, indent=2))
    print("-" * 100 + "\n")


def parse_ts_from_img_file_name(name: str) -> np.int64:
    if not name.endswith(".jpg"):
        raise ValueError(f"Expected .jpg image, got: {name}")

    parts = name[:-4].split("_")
    return np.int64(int(parts[-2]) * 1_000_000_000 + int(parts[-1]))


def get_img_ts_np(img_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    file_names = np.array([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    if len(file_names) == 0:
        raise RuntimeError(f"No .jpg images found in: {img_dir}")

    ts_unsorted = np.fromiter(
        (parse_ts_from_img_file_name(f) for f in file_names),
        dtype=np.int64,
        count=len(file_names),
    )

    sort_idx = np.argsort(ts_unsorted, kind="stable")

    return (
        ts_unsorted[sort_idx],
        np.array([str(img_dir / f) for f in file_names[sort_idx]]),
    )


def is_sorted_ascending(arr: np.ndarray) -> bool:
    return bool(np.all(arr[:-1] <= arr[1:]))


def closest_indices_with_threshold(
    anchor_ts: np.ndarray,
    query_ts: np.ndarray,
    max_diff_ns: int,
) -> np.ndarray:
    if len(query_ts) == 0:
        return np.full(len(anchor_ts), -1, dtype=np.int64)

    insert_idx = np.searchsorted(query_ts, anchor_ts)

    left_idx = np.clip(insert_idx - 1, 0, len(query_ts) - 1)
    right_idx = np.clip(insert_idx, 0, len(query_ts) - 1)

    left_diff = np.abs(anchor_ts - query_ts[left_idx])
    right_diff = np.abs(anchor_ts - query_ts[right_idx])

    use_right = right_diff < left_diff
    nearest_idx = np.where(use_right, right_idx, left_idx)
    nearest_diff = np.where(use_right, right_diff, left_diff)

    nearest_idx[nearest_diff > max_diff_ns] = -1
    return nearest_idx


def timestamp_column_to_ns(df: pd.DataFrame) -> np.ndarray:
    ts = df["timestamp"]

    if np.issubdtype(ts.dtype, np.floating):
        return np.round(ts.to_numpy() * 1e9).astype(np.int64)

    if np.issubdtype(ts.dtype, np.integer):
        return ts.to_numpy(dtype=np.int64)

    raise TypeError(f"Unexpected timestamp dtype: {ts.dtype}")


def sync_via_ts(ep_dir: Path):
    kinematics_df = pd.read_csv(ep_dir / constants.EPISODE_CSV_FILENAME)

    left_endo_ts, left_endo_paths = get_img_ts_np(
        ep_dir / constants.LEFT_ENDOSCOPE_CAM_DIR_NAME
    )
    right_endo_ts, right_endo_paths = get_img_ts_np(
        ep_dir / constants.RIGHT_ENDOSCOPE_CAM_DIR_NAME
    )
    left_wrist_ts, left_wrist_paths = get_img_ts_np(
        ep_dir / constants.PSM2_WRIST_CAM_DIR_NAME
    )
    right_wrist_ts, right_wrist_paths = get_img_ts_np(
        ep_dir / constants.PSM1_WRIST_CAM_DIR_NAME
    )

    kinematics_ts = timestamp_column_to_ns(kinematics_df)

    arrays = [
        left_endo_ts,
        right_endo_ts,
        left_wrist_ts,
        right_wrist_ts,
        kinematics_ts,
    ]

    if not all(is_sorted_ascending(arr) for arr in arrays):
        raise RuntimeError(f"One or more timestamp arrays are not sorted: {ep_dir}")

    max_diff_ns = int(1e9 / constants.FPS)
    left_endo_idx = np.arange(len(left_endo_ts), dtype=np.int64)

    index_map = np.column_stack(
        [
            left_endo_idx,
            closest_indices_with_threshold(left_endo_ts, right_endo_ts, max_diff_ns),
            closest_indices_with_threshold(left_endo_ts, kinematics_ts, max_diff_ns),
            closest_indices_with_threshold(left_endo_ts, left_wrist_ts, max_diff_ns),
            closest_indices_with_threshold(left_endo_ts, right_wrist_ts, max_diff_ns),
        ]
    )

    index_map = index_map[np.all(index_map[:, 1:] != -1, axis=1)]

    if len(index_map) == 0:
        raise RuntimeError(f"No synced frames found for episode: {ep_dir}")

    return (
        left_endo_paths[index_map[:, 0]],
        right_endo_paths[index_map[:, 1]],
        kinematics_df.iloc[index_map[:, 2]].reset_index(drop=True),
        left_wrist_paths[index_map[:, 3]],
        right_wrist_paths[index_map[:, 4]],
    )


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_frame_images(paths):
    left_endo_path, right_endo_path, left_wrist_path, right_wrist_path = paths

    return {
        "images.endoscope.left": load_rgb(left_endo_path),
        "images.endoscope.right": load_rgb(right_endo_path),
        "images.wrist.left": load_rgb(left_wrist_path),
        "images.wrist.right": load_rgb(right_wrist_path),
    }


def patch_fast_image_handling(lerobot_dataset):
    """
    Allows pre-placed .jpg hardlinks to be reused by LeRobot instead of decoding
    source JPEGs and re-saving intermediate images.
    """
    try:
        import lerobot.datasets.image_writer as iw
    except Exception:
        iw = None

    original_get_image_file_path = lerobot_dataset._get_image_file_path
    original_save_image = lerobot_dataset._save_image

    def _get_image_file_path_jpg(
        self,
        episode_index: int,
        image_key: str,
        frame_index: int,
    ) -> Path:
        return original_get_image_file_path(
            episode_index=episode_index,
            image_key=image_key,
            frame_index=frame_index,
        )

    def _save_image_skip_existing(self, image, fpath: Path, compress_level=None) -> None:
        fpath = Path(fpath)
        if fpath.exists():
            return

        if compress_level is None:
            original_save_image(image, fpath)
        else:
            original_save_image(image, fpath, compress_level)

    lerobot_dataset._get_image_file_path = types.MethodType(
        _get_image_file_path_jpg,
        lerobot_dataset,
    )
    lerobot_dataset._save_image = types.MethodType(
        _save_image_skip_existing,
        lerobot_dataset,
    )

    if lerobot_dataset.image_writer is not None and hasattr(
        lerobot_dataset.image_writer,
        "queue",
    ):
        lerobot_dataset.image_writer.queue.maxsize = 128

    if iw is not None:
        original_write_image = iw.write_image

        def _write_image_skip_existing(image, fpath: Path):
            fpath = Path(fpath)
            if fpath.exists():
                return
            original_write_image(image, fpath)

        iw.write_image = _write_image_skip_existing


def hardlink_or_copy(src: str, dst: Path, use_hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if use_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def preplace_episode_images(
    lerobot_dataset,
    episode_index: int,
    camera_paths: dict,
    num_workers: int,
    use_hardlink: bool,
):
    jobs = []

    for image_key, paths in camera_paths.items():
        for frame_idx, src in enumerate(paths):
            dst = lerobot_dataset._get_image_file_path(
                episode_index=episode_index,
                image_key=image_key,
                frame_index=frame_idx,
            )
            jobs.append((src, dst))

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(
            pool.map(
                lambda job: hardlink_or_copy(job[0], job[1], use_hardlink),
                jobs,
            )
        )


def iter_lerobot_frames_from_ep(
    ep_dir: Path,
    lerobot_dataset,
    num_workers: int,
    use_preplace: bool,
    use_hardlink: bool,
):
    (
        left_endo_files,
        right_endo_files,
        kinematics,
        left_wrist_files,
        right_wrist_files,
    ) = sync_via_ts(ep_dir)

    low_level_phase = dataset.get_low_level_phase_from_ep_dir(str(ep_dir))
    high_level_phase = dataset.get_high_level_phase_from_ep_dir(str(ep_dir))
    tissue_name = dataset.get_tissue_name_from_ep_dir(str(ep_dir))
    collector = dataset.get_collector_from_ep_dir(str(ep_dir))

    if re.search(r"\d+", str(tissue_name)) is None:
        raise RuntimeError(f"Could not parse tissue id from: {tissue_name}")

    states = np.concatenate(
        [
            kinematics[constants.HEADER_NAME_QPOS_PSM1].to_numpy(dtype=np.float32),
            kinematics[constants.HEADER_NAME_QPOS_PSM2].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

    actions = np.concatenate(
        [
            kinematics[constants.HEADER_NAME_ACTIONS_PSM1].to_numpy(dtype=np.float32),
            kinematics[constants.HEADER_NAME_ACTIONS_PSM2].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

    esu_signal = kinematics[constants.HEADER_NAME_ESU_SIGNAL].to_numpy()

    camera_paths = {
        "images.endoscope.left": left_endo_files,
        "images.endoscope.right": right_endo_files,
        "images.wrist.left": left_wrist_files,
        "images.wrist.right": right_wrist_files,
    }

    static_metadata = {
        "meta.tissue_id": str(tissue_name),
        "task": str(low_level_phase),
        "meta.low_level_phase": str(low_level_phase),
        "meta.high_level_phase": str(high_level_phase),
        "meta.data_collector": str(collector),
        "meta.tool.psm1": "cautery_hook",
        "meta.tool.psm2": "prograsp",
    }

    if use_preplace:
        episode_index = lerobot_dataset.meta.total_episodes
        preplace_episode_images(
            lerobot_dataset=lerobot_dataset,
            episode_index=episode_index,
            camera_paths=camera_paths,
            num_workers=num_workers,
            use_hardlink=use_hardlink,
        )

        dummy_images = {
            key: np.zeros(tuple(lerobot_dataset.features[key]["shape"]), dtype=np.uint8)
            for key in camera_paths
        }

        for frame_idx in range(len(left_endo_files)):
            yield {
                "state": states[frame_idx],
                "action": actions[frame_idx],
                "esu_signal": str(esu_signal[frame_idx]),
                **static_metadata,
                **dummy_images,
            }

    else:
        image_path_groups = zip(
            left_endo_files,
            right_endo_files,
            left_wrist_files,
            right_wrist_files,
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for frame_idx, image_dict in enumerate(
                executor.map(load_frame_images, image_path_groups)
            ):
                yield {
                    "state": states[frame_idx],
                    "action": actions[frame_idx],
                    "esu_signal": str(esu_signal[frame_idx]),
                    **static_metadata,
                    **image_dict,
                }


def create_lerobot_dataset(args):
    kwargs = dict(
        repo_id=args.repo_id,
        robot_type=constants.ROBOT_NAME,
        fps=constants.FPS,
        features=constants.CHOLECYSTECTOMY_FEATURES,
        image_writer_threads=args.num_workers,
        image_writer_processes=0,
        batch_encoding_size=args.batch_encoding_size,
    )

    try:
        return LeRobotDataset.create(**kwargs)
    except TypeError:
        # For older/newer LeRobot versions that do not support every kwarg.
        kwargs.pop("image_writer_processes", None)
        kwargs.pop("batch_encoding_size", None)
        try:
            return LeRobotDataset.create(**kwargs)
        except TypeError:
            kwargs.pop("image_writer_threads", None)
            return LeRobotDataset.create(**kwargs)


def main():
    args = parse_args()

    if args.overwrite:
        delete_hf_dataset(args.repo_id)

    lerobot_dataset = create_lerobot_dataset(args)

    use_preplace = not args.disable_preplace
    if use_preplace:
        patch_fast_image_handling(lerobot_dataset)

    ep_dirs, info = dataset.get_all_episode_directories(args.source_dir)
    print_ep_info(info)

    for ep_dir in tqdm(ep_dirs, desc="Converting episodes"):
        try:
            ep_dir = Path(ep_dir)

            for frame in iter_lerobot_frames_from_ep(
                ep_dir=ep_dir,
                lerobot_dataset=lerobot_dataset,
                num_workers=args.num_workers,
                use_preplace=use_preplace,
                use_hardlink=not args.no_hardlink,
            ):
                lerobot_dataset.add_frame(frame)

            lerobot_dataset.save_episode()
        except Exception as e:
            print(f"Skipping episode directory {ep_dir} due to error: {e}...")

    if hasattr(lerobot_dataset, "finalize"):
        lerobot_dataset.finalize()


if __name__ == "__main__":
    main()