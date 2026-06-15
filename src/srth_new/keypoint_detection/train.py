import argparse
from glob import glob
import json
import os
from pathlib import Path
import random

import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

from srth_new.general import constants

VAL_PROPORTION = 0.2
DEBUG_SAMPLE_COUNT = 20
DEBUG_RANDOM_SEED = 0

YOLO_KP_DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "keypoint_affordance_dataset"
YOLO_KP_DATASET_IMAGES_TRAIN = YOLO_KP_DATASET_ROOT / "images" / "train"
YOLO_KP_DATASET_IMAGES_VAL = YOLO_KP_DATASET_ROOT / "images" / "val"
YOLO_KP_DATASET_LABELS_TRAIN = YOLO_KP_DATASET_ROOT / "labels" / "train"
YOLO_KP_DATASET_LABELS_VAL = YOLO_KP_DATASET_ROOT / "labels" / "val"
YOLO_KP_DATASET_META = YOLO_KP_DATASET_ROOT / "meta.json"
YOLO_KP_DATASET_YAML = YOLO_KP_DATASET_ROOT / "dataset.yaml"
YOLO_KP_DATASET_DEBUG_DIR = YOLO_KP_DATASET_ROOT / "debug"

CLASS_NAMES = {
    0: "affordance_kp",
    1: "tool_kp",
}

# Your writer appends visibility, so this must be [1, 3].
# If you truly want [1, 2], remove the vis value everywhere below.
KPT_SHAPE = [1, 3]


def create_yolo_kp_dataset_structure():
    YOLO_KP_DATASET_IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    YOLO_KP_DATASET_IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    YOLO_KP_DATASET_LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
    YOLO_KP_DATASET_LABELS_VAL.mkdir(parents=True, exist_ok=True)
    YOLO_KP_DATASET_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    yaml_dict = {
        "path": str(YOLO_KP_DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "affordance_kp",
            1: "tool_kp",
        },
        "kpt_shape": KPT_SHAPE,
    }

    with open(YOLO_KP_DATASET_YAML, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)


def get_images_in_yolo_kp_dataset():
    if not os.path.exists(YOLO_KP_DATASET_META):
        return set()
    with open(YOLO_KP_DATASET_META, "r") as file:
        meta_dict = json.load(file)
    return set(meta_dict["image_symlink_map"].values())


def get_valid_ep_dirs_from_verified_annotations():
    verified_ann_json_paths = glob(
        str(
            constants.VERIFIED_ANNOTATION_OUTPUT_DIR /
            constants.FINAL_VERIFIED_ANNOTATION_SUBDIR /
            "*.json"
        )
    )

    valid_ep_dirs = []
    for json_path in verified_ann_json_paths:
        with open(json_path, "r") as file:
            ann_dict = json.load(file)

        valid_ep_dirs.append(
            (
                str(Path(ann_dict["image_path"]).parent.parent),
                ann_dict[constants.PROPAGATED_POINTS_DICT_KEY],
            )
        )

    return valid_ep_dirs


def _point_to_pixels(x: float, y: float, img_w: int, img_h: int):
    """
    Accept either pixel coords or normalized coords.
    If values look normalized, convert them to pixels.
    """
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * img_w, y * img_h
    return x, y


def _normalize_xy(x_px: float, y_px: float, img_w: int, img_h: int):
    return x_px / img_w, y_px / img_h


def _normalize_bbox_xywh(cx_px: float, cy_px: float, w_px: float, h_px: float, img_w: int, img_h: int):
    return cx_px / img_w, cy_px / img_h, w_px / img_w, h_px / img_h


def _fallback_bbox_from_point(kp_xy_px, img_w: int, img_h: int, side_px: float = 40.0):
    """
    Ultralytics pose labels require a bbox even if your source annotation is only a point.
    This box is a small placeholder centered on the point.
    Replace with a true bbox if you have one.
    """
    x = float(kp_xy_px[0])
    y = float(kp_xy_px[1])
    half = side_px / 2.0

    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(img_w), x + half)
    y2 = min(float(img_h), y + half)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return cx, cy, w, h


def _make_pose_row(cls_idx: int, kp_xy, img_w: int, img_h: int, vis: int = 2):
    """
    Create one Ultralytics pose label row:
    class xc yc w h kx ky vis
    """
    kp_x_px, kp_y_px = _point_to_pixels(float(kp_xy[0]), float(kp_xy[1]), img_w, img_h)

    cx_px, cy_px, w_px, h_px = _fallback_bbox_from_point((kp_x_px, kp_y_px), img_w, img_h)
    bx, by, bw, bh = _normalize_bbox_xywh(cx_px, cy_px, w_px, h_px, img_w, img_h)
    kx, ky = _normalize_xy(kp_x_px, kp_y_px, img_w, img_h)

    return f"{cls_idx} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {kx:.6f} {ky:.6f} {int(vis)}"


def create_data_sample(new_img_name: str, raw_dataset_img_path: Path, kpts: torch.Tensor):
    if random.random() < VAL_PROPORTION:
        new_img_parent_dir = YOLO_KP_DATASET_IMAGES_VAL
        new_label_parent_dir = YOLO_KP_DATASET_LABELS_VAL
    else:
        new_img_parent_dir = YOLO_KP_DATASET_IMAGES_TRAIN
        new_label_parent_dir = YOLO_KP_DATASET_LABELS_TRAIN

    new_img_full_path = new_img_parent_dir / new_img_name
    if new_img_full_path.exists():
        raise FileExistsError(
            f"Image already exists: {new_img_full_path}. "
            "Each newly synced image must have a unique name."
        )

    # Symlink the original image into the YOLO dataset.
    new_img_full_path.symlink_to(raw_dataset_img_path)

    new_img_ann_txt_path = new_label_parent_dir / Path(new_img_name).with_suffix(".txt").name

    with Image.open(raw_dataset_img_path) as im:
        img_w, img_h = im.size

    a = constants.AFFORDANCE_KP_CLS
    b = constants.TOOL_KP_CLS

    # Expecting one point per class.
    annotation_lines = [
        _make_pose_row(a, kpts[a][:2], img_w, img_h, vis=2),
        _make_pose_row(b, kpts[b][:2], img_w, img_h, vis=2),
    ]

    with open(new_img_ann_txt_path, "w") as file:
        file.write("\n".join(annotation_lines) + "\n")


def load_tensor(bin_path):
    tensor = torch.load(bin_path, map_location="cpu")
    if isinstance(tensor, dict):
        raise ValueError(
            "Expected a tensor, but got a dict. Check how the .bin file was saved."
        )
    return tensor


def sync_yolo_kp_dataset_to_annotations():
    yolo_kp_orig_img_paths = get_images_in_yolo_kp_dataset()
    valid_ep_dirs = get_valid_ep_dirs_from_verified_annotations()

    added_images = {}

    for valid_ep_dir, kp_tensor_path in tqdm(valid_ep_dirs, desc="Syncing annotations..."):
        valid_images = glob(str(Path(valid_ep_dir) / constants.LEFT_ENDOSCOPE_CAM_DIR_NAME / "*.jpg"))
        kp_tensor = load_tensor(kp_tensor_path).cpu()[0]

        for idx, valid_img in enumerate(valid_images):
            if valid_img not in yolo_kp_orig_img_paths:
                new_img_name = f"img_{len(yolo_kp_orig_img_paths) + len(added_images):09d}.jpg"
                image_frame_idx = int(str(Path(valid_img).name)[5:11])
                create_data_sample(new_img_name, Path(valid_img), kp_tensor[image_frame_idx])
                added_images[new_img_name] = valid_img

    if not os.path.exists(YOLO_KP_DATASET_META):
        meta_dict = {"image_symlink_map": {}}
    else:
        with open(YOLO_KP_DATASET_META, "r") as file:
            meta_dict = json.load(file)

    overlap = set(meta_dict["image_symlink_map"]) & set(added_images)
    if overlap:
        raise ValueError(f"Duplicate keys found: {overlap}")

    meta_dict["image_symlink_map"] = meta_dict["image_symlink_map"] | added_images

    with open(YOLO_KP_DATASET_META, "w") as file:
        json.dump(meta_dict, file, indent=2)


def parse_pose_label_line(line: str):
    vals = line.strip().split()
    if len(vals) < 8:
        raise ValueError(f"Bad label line: {line}")

    cls = int(vals[0])
    xc, yc, bw, bh = map(float, vals[1:5])

    # For this dataset we expect exactly one keypoint with visibility.
    kx, ky, vis = map(float, vals[5:8])
    return cls, xc, yc, bw, bh, [(kx, ky, int(vis))]


def visualize_image(image_path: Path, label_path: Path, save_path: Path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis("off")

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for line in lines:
                cls, xc, yc, bw, bh, kpts = parse_pose_label_line(line)

                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                rect_w = bw * img_w
                rect_h = bh * img_h

                rect = plt.Rectangle(
                    (x1, y1),
                    rect_w,
                    rect_h,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax.add_patch(rect)

                ax.text(
                    x1,
                    max(0, y1 - 5),
                    CLASS_NAMES.get(cls, str(cls)),
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2),
                )

                for kx, ky, vis in kpts:
                    px = kx * img_w
                    py = ky * img_h
                    if vis > 0:
                        ax.plot(px, py, "ro", markersize=6)
                        ax.text(
                            px + 4,
                            py + 4,
                            f"v={vis}",
                            color="yellow",
                            fontsize=8,
                        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_random_debug_samples(n=DEBUG_SAMPLE_COUNT):
    random.seed(DEBUG_RANDOM_SEED)

    image_paths = sorted(list(YOLO_KP_DATASET_IMAGES_VAL.glob("*.jpg")))
    if not image_paths:
        raise RuntimeError(f"No validation images found in {YOLO_KP_DATASET_IMAGES_VAL}")

    sampled_paths = random.sample(image_paths, min(n, len(image_paths)))

    for image_path in sampled_paths:
        label_path = YOLO_KP_DATASET_LABELS_VAL / image_path.with_suffix(".txt").name
        save_path = YOLO_KP_DATASET_DEBUG_DIR / f"{image_path.stem}_viz.png"
        visualize_image(image_path, label_path, save_path)
        print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate random visualization samples in the debug directory.",
    )

    args = parser.parse_args()

    if not os.path.isdir(YOLO_KP_DATASET_ROOT):
        create_yolo_kp_dataset_structure()
    else:
        YOLO_KP_DATASET_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    sync_yolo_kp_dataset_to_annotations()

    if args.debug:
        save_random_debug_samples(n=20)

    model = YOLO("yolo26n-pose.pt")  # pretrained pose model
    results = model.train(
        data=str(YOLO_KP_DATASET_ROOT / "dataset.yaml"),
        epochs=1,
        imgsz=640,
        device=[0, 1]
    )
    print(results)


if __name__ == "__main__":
    main()