import os

# Helps with CUDA allocator fragmentation in some workloads.
# Must be set before the first CUDA allocation for best effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import re
import tempfile
from glob import glob
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path

from srth_new.general.constants import (
    COTRACKER_CHECKPOINT_DIR,
    RAW_ANNOTATION_OUTPUT_DIR,
    PROPAGATED_POINTS_DICT_KEY,
    ANNOTATIONS_SUBDIR_NAME,
)

saved_annotation_dir = os.path.join(RAW_ANNOTATION_OUTPUT_DIR, ANNOTATIONS_SUBDIR_NAME)


def natural_key(path):
    """
    Sort frame2.png before frame10.png.
    """
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", Path(path).name)
    ]


def images_to_mp4(
    image_dir,
    output_video,
    fps=30,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
):
    """
    Convert a directory of images into an MP4 video.
    """
    image_dir = Path(image_dir)
    output_video = Path(output_video)

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    images.sort(key=natural_key)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Could not read {images[0]}")

    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    try:
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved video: {output_video}")


def clear_cuda_memory():
    """
    Best-effort GPU memory cleanup between videos.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Load the CoTracker model once.
model = CoTrackerPredictor(
    checkpoint=os.path.join(COTRACKER_CHECKPOINT_DIR, "scaled_offline.pth")
).cuda()
model.eval()

ann_json_files = glob(os.path.join(saved_annotation_dir, "*.json"))

for ann_json_file in tqdm(ann_json_files):
    propagated_points_file = ann_json_file.replace(".json", "_points.pt")
    if os.path.exists(propagated_points_file):
        print(f"Keypoint Annotation File {propagated_points_file} exists...")
        continue

    output_vid_path = None
    video = None
    queries = None
    pred_tracks = None
    pred_visibility = None

    try:
        with open(ann_json_file, "r") as file:
            ann_data = json.load(file)

        image_dir = Path(ann_data["image_path"]).parent
        affordance_kp = ann_data["affordance_xy"]
        tool_tip_kp = ann_data["tool_tip_xy"]
        frame_idx = ann_data["frame_idx"]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_vid_path = Path(tmp.name)

        images_to_mp4(
            image_dir=str(image_dir),
            output_video=output_vid_path,
            fps=30,
        )

        video_np = read_video_from_path(output_vid_path)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().cuda()

        queries = torch.tensor(
            [[
                [frame_idx, affordance_kp[0], affordance_kp[1]],
                [frame_idx, tool_tip_kp[0], tool_tip_kp[1]],
            ]],
            dtype=torch.float32,
            device=video.device,
        )

        with torch.inference_mode():
            pred_tracks, pred_visibility = model(
                video,
                queries=queries,
                backward_tracking=True,
            )

        torch.save(pred_tracks.cpu(), propagated_points_file)

        ann_data[PROPAGATED_POINTS_DICT_KEY] = propagated_points_file
        with open(ann_json_file, "w") as file:
            json.dump(ann_data, file, indent=2)

    finally:
        # Drop references before clearing cache.
        del video, queries, pred_tracks, pred_visibility

        if output_vid_path is not None and output_vid_path.exists():
            try:
                output_vid_path.unlink()
            except OSError:
                pass

        clear_cuda_memory()