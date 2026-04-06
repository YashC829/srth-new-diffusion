from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _episode_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.split("_")[-1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (0, path.name)


def discover_episode_paths(dataset_dir: Path, max_episodes: int | None = None) -> list[Path]:
    episode_paths = sorted(dataset_dir.glob("episode_*.hdf5"), key=_episode_sort_key)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {dataset_dir}.")
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
    return episode_paths


def split_episode_paths(
    episode_paths: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(episode_paths) == 1:
        return episode_paths, episode_paths

    rng = np.random.default_rng(seed)
    shuffled = [episode_paths[idx] for idx in rng.permutation(len(episode_paths))]
    num_val = int(round(len(shuffled) * val_ratio))
    num_val = max(1, num_val)
    num_val = min(num_val, len(shuffled) - 1)

    val_paths = shuffled[:num_val]
    train_paths = shuffled[num_val:]
    return train_paths, val_paths


def compute_norm_stats(episode_paths: list[Path]) -> dict[str, np.ndarray]:
    qpos_chunks = []
    action_chunks = []

    for episode_path in episode_paths:
        with h5py.File(episode_path, "r") as root:
            qpos_chunks.append(root["/observations/qpos"][()].astype(np.float32))
            action_chunks.append(root["/action"][()].astype(np.float32))

    all_qpos = np.concatenate(qpos_chunks, axis=0)
    all_actions = np.concatenate(action_chunks, axis=0)

    qpos_std = np.clip(all_qpos.std(axis=0).astype(np.float32), 1e-2, np.inf)
    action_std = np.clip(all_actions.std(axis=0).astype(np.float32), 1e-2, np.inf)

    return {
        "qpos_mean": all_qpos.mean(axis=0).astype(np.float32),
        "qpos_std": qpos_std,
        "action_mean": all_actions.mean(axis=0).astype(np.float32),
        "action_std": action_std,
    }


def save_stats(stats: dict[str, np.ndarray], stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("wb") as handle:
        pickle.dump(stats, handle)


def load_stats(stats_path: Path) -> dict[str, np.ndarray]:
    with stats_path.open("rb") as handle:
        return pickle.load(handle)


class EpisodicDataset(Dataset):
    def __init__(
        self,
        episode_paths: list[Path],
        camera_names: list[str],
        stats: dict[str, np.ndarray],
        chunk_size: int,
        train: bool,
    ) -> None:
        self.episode_paths = episode_paths
        self.camera_names = camera_names
        self.stats = stats
        self.chunk_size = chunk_size
        self.train = train

    def __len__(self) -> int:
        return len(self.episode_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_path = self.episode_paths[index]
        with h5py.File(episode_path, "r") as root:
            action_dataset = root["/action"]
            num_steps = action_dataset.shape[0]
            start_ts = self._sample_start_timestep(num_steps)
            end_ts = min(num_steps, start_ts + self.chunk_size)

            qpos = root["/observations/qpos"][start_ts].astype(np.float32)
            actions = action_dataset[start_ts:end_ts].astype(np.float32)
            images = self._load_images(root, start_ts)

        padded_actions = np.zeros((self.chunk_size, actions.shape[-1]), dtype=np.float32)
        padded_actions[: actions.shape[0]] = actions
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[: actions.shape[0]] = False

        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded_actions = (padded_actions - self.stats["action_mean"]) / self.stats["action_std"]
        image_array = np.stack(images, axis=0).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (0, 3, 1, 2))

        return (
            torch.from_numpy(image_array),
            torch.from_numpy(qpos.astype(np.float32)),
            torch.from_numpy(padded_actions.astype(np.float32)),
            torch.from_numpy(is_pad),
        )

    def _sample_start_timestep(self, num_steps: int) -> int:
        if num_steps <= 1:
            return 0
        if self.train:
            return int(np.random.randint(0, num_steps))
        return max(0, (num_steps - self.chunk_size) // 2)

    def _load_images(self, root: h5py.File, timestep: int) -> list[np.ndarray]:
        compressed = bool(root.attrs.get("compress", False))
        images = []

        for camera_name in self.camera_names:
            image = root[f"/observations/images/{camera_name}"][timestep]
            if compressed:
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        return images


def build_dataloader(
    episode_paths: list[Path],
    camera_names: list[str],
    stats: dict[str, np.ndarray],
    chunk_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    dataset = EpisodicDataset(
        episode_paths=episode_paths,
        camera_names=camera_names,
        stats=stats,
        chunk_size=chunk_size,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
