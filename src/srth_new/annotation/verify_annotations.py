import json
import os
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from glob import glob

import cv2
import torch
from tqdm import tqdm

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from srth_new.general import constants


def get_raw_ann_dict_paths():
    return glob(
        os.path.join(
            constants.RAW_ANNOTATION_OUTPUT_DIR,
            constants.ANNOTATIONS_SUBDIR_NAME,
            "*.json",
        )
    )


def load_tensor(bin_path):
    tensor = torch.load(bin_path, map_location="cpu")
    if isinstance(tensor, dict):
        raise ValueError(
            "Expected a tensor, but got a dict. Check how the .bin file was saved."
        )
    return tensor


def get_image_paths(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    return sorted(paths)


def stable_id_from_path(path_str: str, n: int = 16) -> str:
    resolved = str(Path(path_str).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:n]


def verification_paths_for_raw_ann(raw_ann_dict_path: str):
    raw_path = Path(raw_ann_dict_path)
    uid = stable_id_from_path(raw_ann_dict_path)

    video_name = f"{raw_path.stem}_{uid}.mp4"
    json_name = f"{raw_path.stem}_{uid}.json"

    video_path = (
        Path(constants.VERIFIED_ANNOTATION_OUTPUT_DIR)
        / constants.VERIFICATION_VIDEO_SUBDIR
        / video_name
    )
    ann_path = (
        Path(constants.VERIFIED_ANNOTATION_OUTPUT_DIR)
        / constants.VERIFICATION_ANNOTATION_JSON_SUBDIR
        / json_name
    )
    return video_path, ann_path


def get_verified_annotation_subdir() -> Path:
    subdir = getattr(constants, "VERIFIED_ANNOTATION_SUBDIR", "annotations")
    return Path(constants.VERIFIED_ANNOTATION_OUTPUT_DIR) / subdir


def save_annotated_video(
    image_dir,
    bin_path,
    output_video_path,
    fps=30,
    point_radius=6,
):
    tensor = load_tensor(bin_path)

    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2:] != (2, 2):
        raise ValueError(
            f"Expected tensor shape [1, frames, 2, 2], got {tuple(tensor.shape)}"
        )

    tensor = tensor[0]
    image_paths = get_image_paths(image_dir)

    num_frames = min(len(image_paths), tensor.shape[0])
    if num_frames == 0:
        raise ValueError("No matching images or tensor frames found.")

    colors = [(0, 0, 255), (0, 255, 0)]

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = first_img.shape[:2]

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video_path}")

    try:
        for i in range(num_frames):
            img_path = image_paths[i]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))

            frame_pts = tensor[i]  # [2, 2]

            for j in range(frame_pts.shape[0]):
                x, y = frame_pts[j].tolist()

                # If coordinates are normalized, convert here.
                # x = x * width
                # y = y * height

                x = int(round(x))
                y = int(round(y))

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img, (x, y), point_radius, colors[j % len(colors)], -1)

            writer.write(img)
    finally:
        writer.release()

    print(f"Saved video to: {output_video_path}")


def load_raw_annotation(raw_ann_dict_path: str) -> dict | None:
    try:
        with open(raw_ann_dict_path, "r") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        print(f"[WARNING] Could not load json file {raw_ann_dict_path}. Skipping...")
        return None


def remove_sequence_path_from_list(sequence_path: str):
    txt_path = Path(constants.RAW_ANNOTATION_OUTPUT_DIR) / "annotated_sequences.txt"
    if not txt_path.exists():
        return

    try:
        target_norm = os.path.normpath(str(Path(sequence_path).resolve()))
    except Exception:
        target_norm = os.path.normpath(str(sequence_path))

    kept_lines = []
    with open(txt_path, "r") as file:
        for line in file:
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                line_norm = os.path.normpath(str(Path(raw_line).resolve()))
            except Exception:
                line_norm = os.path.normpath(raw_line)

            if line_norm != target_norm:
                kept_lines.append(line.rstrip("\n"))

    with open(txt_path, "w") as file:
        file.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))


def safe_delete(path_like):
    if not path_like:
        return
    path = Path(path_like)
    if path.exists():
        path.unlink()


def safe_move_file(src_path, dst_dir: Path) -> Path | None:
    src = Path(src_path)
    if not src.exists():
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        dst.unlink()

    shutil.move(str(src), str(dst))
    return dst


def accept_annotation(raw_ann_dict_path: str, raw_ann_dict: dict):
    verified_dir = get_verified_annotation_subdir()
    verified_dir.mkdir(parents=True, exist_ok=True)

    propagated_key = constants.PROPAGATED_POINTS_DICT_KEY
    propagated_src = raw_ann_dict.get(propagated_key)

    # Move the raw annotation JSON into the verified folder.
    new_raw_ann_path = safe_move_file(raw_ann_dict_path, verified_dir)
    if new_raw_ann_path is None:
        raise FileNotFoundError(f"Could not move raw annotation file: {raw_ann_dict_path}")

    # Move the propagated tensor/bin file into the verified folder.
    new_propagated_path = None
    if propagated_src:
        new_propagated_path = safe_move_file(propagated_src, verified_dir)

    # Update the copied JSON so it points to the new propagated file location.
    try:
        with open(new_raw_ann_path, "r") as file:
            moved_ann = json.load(file)
    except (OSError, json.JSONDecodeError):
        moved_ann = dict(raw_ann_dict)

    if new_propagated_path is not None:
        moved_ann[propagated_key] = str(new_propagated_path)
        with open(new_raw_ann_path, "w") as file:
            json.dump(moved_ann, file, indent=3)


def reject_annotation(raw_ann_dict_path: str, raw_ann_dict: dict):
    propagated_key = constants.PROPAGATED_POINTS_DICT_KEY
    propagated_src = raw_ann_dict.get(propagated_key)

    safe_delete(raw_ann_dict_path)
    safe_delete(propagated_src)

    remove_sequence_path_from_list(str(Path(raw_ann_dict["image_path"]).parent))


@dataclass
class ReviewItem:
    raw_ann_dict_path: str
    verification_video_path: Path
    verification_ann_path: Path
    raw_ann_dict: dict


class AnnotationReviewGUI:
    def __init__(self, review_items: list[ReviewItem]):
        self.review_items = review_items
        self.index = 0
        self.cap = None
        self.current_frame_index = 0
        self.frame_count = 0
        self.current_video_path = None

        self.root = tk.Tk()
        self.root.title("Annotation Review")
        self.root.geometry("1200x900")

        self.video_label = tk.Label(self.root, text="Loading...")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.status_label = tk.Label(self.root, text="", anchor="w", justify="left")
        self.status_label.pack(fill="x", padx=10, pady=(0, 8))

        self.frame_slider = tk.Scale(
            self.root,
            from_=0,
            to=0,
            orient="horizontal",
            resolution=1,
            command=self.on_slider_change,
            length=1000,
        )
        self.frame_slider.pack(fill="x", padx=10, pady=(0, 8))

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(0, 10))

        self.accept_button = tk.Button(controls, text="Accept (a)", command=self.accept_current)
        self.accept_button.pack(side="left", padx=(0, 8))

        self.reject_button = tk.Button(controls, text="Reject (s)", command=self.reject_current)
        self.reject_button.pack(side="left", padx=(0, 8))

        self.prev_button = tk.Button(controls, text="Previous", command=self.previous_item)
        self.prev_button.pack(side="left", padx=(0, 8))

        self.next_button = tk.Button(controls, text="Next", command=self.next_item)
        self.next_button.pack(side="left", padx=(0, 8))

        self.root.bind("<KeyPress-a>", lambda _evt: self.accept_current())
        self.root.bind("<KeyPress-A>", lambda _evt: self.accept_current())
        self.root.bind("<KeyPress-s>", lambda _evt: self.reject_current())
        self.root.bind("<KeyPress-S>", lambda _evt: self.reject_current())
        self.root.bind("<Left>", lambda _evt: self.step_frame(-1))
        self.root.bind("<Right>", lambda _evt: self.step_frame(1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_item(0)

    def run(self):
        self.root.mainloop()

    def on_close(self):
        self.close_video()
        self.root.destroy()

    def close_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_video(self, video_path: Path):
        self.close_video()
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_slider.configure(to=max(0, self.frame_count - 1))
        self.current_frame_index = 0
        self.frame_slider.set(0)

    def load_item(self, index: int):
        if not self.review_items:
            messagebox.showinfo("Annotation Review", "No pending annotations to review.")
            self.on_close()
            return

        self.index = max(0, min(index, len(self.review_items) - 1))
        item = self.review_items[self.index]

        try:
            self.open_video(item.verification_video_path)
        except Exception as exc:
            messagebox.showerror("Video Error", str(exc))
            self.remove_current_item(delete_files=False)
            return

        self.update_status()
        self.show_frame(0)

    def update_status(self):
        item = self.review_items[self.index]
        self.status_label.config(
            text=(
                f"Item {self.index + 1} / {len(self.review_items)}\n"
                f"Raw annotation: {item.raw_ann_dict_path}\n"
                f"Video: {item.verification_video_path}\n"
                f"Frame: {self.current_frame_index + 1} / {max(self.frame_count, 1)}"
            )
        )

    def on_slider_change(self, value):
        try:
            frame_idx = int(float(value))
        except ValueError:
            return
        self.show_frame(frame_idx)

    def step_frame(self, delta: int):
        if self.frame_count <= 0:
            return
        new_idx = max(0, min(self.current_frame_index + delta, self.frame_count - 1))
        self.frame_slider.set(new_idx)
        self.show_frame(new_idx)

    def show_frame(self, frame_idx: int):
        if self.cap is None or self.frame_count <= 0:
            return

        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        self.current_frame_index = frame_idx

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        max_w = 1100
        max_h = 700
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")
        self.update_status()

    def accept_current(self):
        self.apply_action("accept")

    def reject_current(self):
        self.apply_action("reject")

    def apply_action(self, action: str):
        if not self.review_items:
            return

        item = self.review_items[self.index]

        try:
            if action == "accept":
                accept_annotation(item.raw_ann_dict_path, item.raw_ann_dict)
            elif action == "reject":
                reject_annotation(item.raw_ann_dict_path, item.raw_ann_dict)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as exc:
            messagebox.showerror("Action Failed", str(exc))
            return

        # Always delete the generated review artifacts.
        safe_delete(item.verification_video_path)
        safe_delete(item.verification_ann_path)

        self.remove_current_item(delete_files=False)

    def remove_current_item(self, delete_files=False):
        if delete_files:
            item = self.review_items[self.index]
            safe_delete(item.verification_video_path)
            safe_delete(item.verification_ann_path)

        self.review_items.pop(self.index)

        if not self.review_items:
            messagebox.showinfo("Annotation Review", "Finished reviewing all annotations.")
            self.on_close()
            return

        if self.index >= len(self.review_items):
            self.index = len(self.review_items) - 1

        self.load_item(self.index)

    def next_item(self):
        if not self.review_items:
            return
        next_index = min(self.index + 1, len(self.review_items) - 1)
        self.load_item(next_index)

    def previous_item(self):
        if not self.review_items:
            return
        prev_index = max(self.index - 1, 0)
        self.load_item(prev_index)


def build_review_queue():
    raw_ann_dict_paths = get_raw_ann_dict_paths()
    review_items: list[ReviewItem] = []

    for raw_ann_dict_path in tqdm(raw_ann_dict_paths, desc="Preparing review items"):
        vid_output_path, verification_ann_path = verification_paths_for_raw_ann(raw_ann_dict_path)

        # Fast skip: if both already exist, just add to the review queue.
        if vid_output_path.exists() and verification_ann_path.exists():
            raw_ann_dict = load_raw_annotation(raw_ann_dict_path)
            if raw_ann_dict is None:
                continue
            review_items.append(
                ReviewItem(
                    raw_ann_dict_path=raw_ann_dict_path,
                    verification_video_path=vid_output_path,
                    verification_ann_path=verification_ann_path,
                    raw_ann_dict=raw_ann_dict,
                )
            )
            continue

        raw_ann_dict = load_raw_annotation(raw_ann_dict_path)
        if raw_ann_dict is None:
            continue

        if constants.PROPAGATED_POINTS_DICT_KEY not in raw_ann_dict:
            print(
                f"[WARNING] Raw annotation {raw_ann_dict_path} has not yet had its keypoints propagated. "
                "Please run python propagate_points.py to propagate all keypoints."
            )
            continue

        image_frame_dir = Path(raw_ann_dict["image_path"]).parent

        # Only generate the video if it is missing.
        if not vid_output_path.exists():
            save_annotated_video(
                image_frame_dir,
                raw_ann_dict[constants.PROPAGATED_POINTS_DICT_KEY],
                vid_output_path,
            )

        # Only write the metadata JSON if it is missing.
        if not verification_ann_path.exists():
            verification_ann_path.parent.mkdir(parents=True, exist_ok=True)
            verification_ann_path_info = {
                "orig_ann_path": raw_ann_dict_path,
                "corresponding_verification_video": str(vid_output_path),
            }
            with open(verification_ann_path, "w") as file:
                json.dump(verification_ann_path_info, file, indent=3)

        review_items.append(
            ReviewItem(
                raw_ann_dict_path=raw_ann_dict_path,
                verification_video_path=vid_output_path,
                verification_ann_path=verification_ann_path,
                raw_ann_dict=raw_ann_dict,
            )
        )

    return review_items


def main():
    review_items = build_review_queue()

    if not review_items:
        print("No pending annotations to review.")
        return

    app = AnnotationReviewGUI(review_items)
    app.run()


if __name__ == "__main__":
    main()