import argparse
import json
import os
import queue
import re
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QListWidget,
    QMessageBox,
    QSizePolicy,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

from srth_new.general import constants


def natural_key(path: Path):
    """
    Sort frame000001 before frame000010.
    """
    text = path.name
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def safe_annotation_name(left_img_dir: Path, dataset_root: Path) -> str:
    """
    Turn a sequence path into a stable JSON filename.

    Example:
      Tissue#7/Grayson/unzipping/4_hook_tissue/20260515.../left_img_dir
    becomes:
      Tissue#7__Grayson__unzipping__4_hook_tissue__20260515...__left_img_dir.json
    """
    rel = left_img_dir.resolve().relative_to(dataset_root.resolve())
    parts = list(rel.parts)
    return "__".join(parts) + ".json"


def get_subphase(left_img_dir: Path) -> str:
    """
    Expected structure:
      Tissue#/Annotator/task/subphase/timestamp/left_img_dir

    left_img_dir.parent is timestamp.
    left_img_dir.parent.parent is subphase.
    """
    return left_img_dir.parent.parent.name


def find_left_img_dirs(dataset_root: Path):
    """
    One-time full scan. This is intentionally simple and robust.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        current = Path(dirpath)
        if current.name == "left_img_dir":
            found.append(str(current.resolve()))
            dirnames[:] = []
    found.sort()
    return found


def load_or_create_sequence_index(dataset_root: Path, output_dir: Path, force_rescan: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "sequence_index.json"

    if index_path.exists() and not force_rescan:
        with index_path.open("r") as f:
            data = json.load(f)
        if Path(data["dataset_root"]).resolve() == dataset_root.resolve():
            return [Path(p) for p in data["left_img_dirs"]]

    left_img_dirs = find_left_img_dirs(dataset_root)
    tmp_path = index_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root.resolve()),
                "left_img_dirs": left_img_dirs,
            },
            f,
            indent=2,
        )
    tmp_path.replace(index_path)
    return [Path(p) for p in left_img_dirs]


def load_annotated_sequences(output_dir: Path):
    annotated_path = output_dir / "annotated_sequences.txt"
    if not annotated_path.exists():
        annotated_path.touch()
        return set()
    with annotated_path.open("r") as f:
        return {line.strip() for line in f if line.strip()}


class AsyncAnnotationWriter:
    """
    Saves annotations on a background thread.

    The GUI enqueues save jobs. Disk writing and annotated_sequences.txt updates
    happen in the background.
    """

    def __init__(self, output_dir: Path, dataset_root: Path):
        self.output_dir = output_dir
        self.dataset_root = dataset_root
        self.annotations_dir = output_dir / constants.ANNOTATIONS_SUBDIR_NAME
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_path = output_dir / "annotated_sequences.txt"
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def save(self, left_img_dir: Path, annotation: dict):
        self.q.put((left_img_dir, annotation))

    def flush(self):
        """
        Wait until all queued save jobs are finished.
        """
        self.q.join()

    def _rewrite_annotated_sequences(self):
        tmp_path = self.annotated_path.with_suffix(".txt.tmp")
        with tmp_path.open("w") as f:
            for seq in sorted({str(p) for p in self.annotated_path.parent.parent.glob("*")}):
                pass

        # Rebuild from the current file contents if possible; the GUI also keeps
        # the authoritative in-memory set and rewrites this file itself when undoing.
        # This method is not used in normal save flow.
        if self.annotated_path.exists():
            with self.annotated_path.open("r") as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            lines = []

        with tmp_path.open("w") as f:
            for line in lines:
                f.write(line + "\n")
        tmp_path.replace(self.annotated_path)

    def _write_annotation_file(self, left_img_dir: Path, annotation: dict):
        out_name = safe_annotation_name(left_img_dir, self.dataset_root)
        out_path = self.annotations_dir / out_name
        tmp_path = out_path.with_suffix(".json.tmp")
        with tmp_path.open("w") as f:
            json.dump(annotation, f, indent=2)
        tmp_path.replace(out_path)

    def _append_annotated_sequence(self, left_img_dir: Path):
        with self.annotated_path.open("a") as f:
            f.write(str(left_img_dir.resolve()) + "\n")

    def _run(self):
        while True:
            left_img_dir, annotation = self.q.get()
            try:
                with self.lock:
                    self._write_annotation_file(left_img_dir, annotation)
                    self._append_annotated_sequence(left_img_dir)
            except Exception as exc:
                print(f"[ERROR] Failed to save annotation for {left_img_dir}: {exc}")
            finally:
                self.q.task_done()


class ImageCanvas(QLabel):
    """
    QLabel-based image canvas that supports:
    - scaled image display
    - click-to-place keypoint
    - drawing keypoint overlay
    - mapping widget coordinates back to original image coordinates
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.original_pixmap = None
        self.display_pixmap = None
        self.image_size_wh = None
        self.keypoints = {
            "affordance": None,
            "tool_tip": None,
        }
        self.active_keypoint_type = "affordance"
        self.on_point_changed = None

    def set_image(self, image_path: Path):
        pix = QPixmap(str(image_path))
        if pix.isNull():
            raise RuntimeError(f"Could not load image: {image_path}")
        self.original_pixmap = pix
        self.image_size_wh = (pix.width(), pix.height())
        self._redraw()

    def set_keypoints(self, keypoints: dict):
        self.keypoints = {
            "affordance": keypoints.get("affordance"),
            "tool_tip": keypoints.get("tool_tip"),
        }
        self._redraw()

    def set_active_keypoint_type(self, keypoint_type: str):
        self.active_keypoint_type = keypoint_type

    def clear_keypoint(self, keypoint_type: str | None = None):
        if keypoint_type is None:
            self.keypoints = {"affordance": None, "tool_tip": None}
        else:
            self.keypoints[keypoint_type] = None
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def mousePressEvent(self, event):
        if self.original_pixmap is None:
            return
        if event.button() != Qt.LeftButton:
            return

        image_xy = self._widget_to_image_xy(event.position())
        if image_xy is None:
            return

        self.keypoints[self.active_keypoint_type] = image_xy
        self._redraw()
        if self.on_point_changed is not None:
            self.on_point_changed(self.active_keypoint_type, image_xy)

    def _redraw(self):
        if self.original_pixmap is None:
            self.clear()
            return

        scaled = self.original_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        canvas = QPixmap(scaled)
        painter = QPainter(canvas)

        sx = scaled.width() / self.original_pixmap.width()
        sy = scaled.height() / self.original_pixmap.height()

        colors = {
            "affordance": QColor(255, 0, 0),
            "tool_tip": QColor(0, 120, 255),
        }

        for keypoint_type, point_xy in self.keypoints.items():
            if point_xy is None:
                continue
            x_img, y_img = point_xy
            x_disp = x_img * sx
            y_disp = y_img * sy
            pen = QPen(colors.get(keypoint_type, QColor(255, 255, 0)))
            pen.setWidth(4)
            painter.setPen(pen)
            radius = 7
            painter.drawEllipse(QPointF(x_disp, y_disp), radius, radius)
            cross = 12
            painter.drawLine(
                QPointF(x_disp - cross, y_disp),
                QPointF(x_disp + cross, y_disp),
            )
            painter.drawLine(
                QPointF(x_disp, y_disp - cross),
                QPointF(x_disp, y_disp + cross),
            )

        painter.end()
        self.display_pixmap = canvas
        self.setPixmap(canvas)

    def _widget_to_image_xy(self, pos: QPointF):
        if self.original_pixmap is None or self.pixmap() is None:
            return None

        label_w = self.width()
        label_h = self.height()
        disp_w = self.pixmap().width()
        disp_h = self.pixmap().height()

        x_offset = (label_w - disp_w) / 2.0
        y_offset = (label_h - disp_h) / 2.0

        x_disp = pos.x() - x_offset
        y_disp = pos.y() - y_offset

        if x_disp < 0 or y_disp < 0 or x_disp >= disp_w or y_disp >= disp_h:
            return None

        x_img = x_disp * self.original_pixmap.width() / disp_w
        y_img = y_disp * self.original_pixmap.height() / disp_h
        return [float(x_img), float(y_img)]


class AnnotatorGUI(QWidget):
    def __init__(self, dataset_root: Path, output_dir: Path, force_rescan: bool = False):
        super().__init__()
        self.dataset_root = dataset_root.resolve()
        self.output_dir = output_dir.resolve()

        self.all_sequences = load_or_create_sequence_index(
            self.dataset_root,
            self.output_dir,
            force_rescan=force_rescan,
        )
        self.annotated_sequences = load_annotated_sequences(self.output_dir)
        self.writer = AsyncAnnotationWriter(self.output_dir, self.dataset_root)

        self.current_sequence = None
        self.current_images = []
        self.current_frame_idx = 0
        self.current_keypoints = {
            "affordance": None,
            "tool_tip": None,
        }
        self.current_keypoint_type = "affordance"

        self.last_saved_sequence = None

        self.setWindowTitle("Affordance Keypoint Annotator")

        self.sequence_filter = QComboBox()
        self.sequence_list = QListWidget()
        self.canvas = ImageCanvas()
        self.canvas.on_point_changed = self._on_point_changed
        self.canvas.set_active_keypoint_type(self.current_keypoint_type)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)

        self.frame_slider.setStyleSheet("""
        QSlider::handle:horizontal {
            width: 24px;
            height: 24px;
            margin: -8px 0;
        }
        """)

        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.frame_label = QLabel("Frame: - / -")
        self.sequence_label = QLabel("No sequence loaded")
        self.point_label = QLabel("Affordance: None | Tool tip: None")

        self.keypoint_type_label = QLabel("Keypoint type")
        self.keypoint_type_combo = QComboBox()
        self.keypoint_type_combo.addItems(["affordance", "tool_tip"])
        self.keypoint_type_combo.currentTextChanged.connect(self._on_keypoint_type_changed)

        self.clear_button = QPushButton("Clear Selected Keypoint")
        self.clear_button.clicked.connect(self._clear_keypoint)

        self.save_button = QPushButton("Save Annotation")
        self.save_button.clicked.connect(self._save_annotation)

        self.undo_button = QPushButton("Undo Last Save")
        self.undo_button.clicked.connect(self._undo_last_save)
        self.undo_button.setEnabled(False)

        self.rescan_button = QPushButton("Rescan Dataset")
        self.rescan_button.clicked.connect(self._rescan_dataset)

        self.save_shortcut = QShortcut(QKeySequence("Space"), self)
        self.save_shortcut.activated.connect(self._save_annotation)

        self._build_layout()
        self._populate_filter()
        self._refresh_sequence_list()
        self.sequence_filter.currentTextChanged.connect(self._refresh_sequence_list)
        self.sequence_list.currentRowChanged.connect(self._on_sequence_selected)

    def _build_layout(self):
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Subphase filter"))
        left_panel.addWidget(self.sequence_filter)
        left_panel.addWidget(QLabel("Unannotated sequences"))
        left_panel.addWidget(self.sequence_list)
        left_panel.addWidget(self.rescan_button)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.sequence_label)
        right_panel.addWidget(self.canvas)

        keypoint_row = QHBoxLayout()
        keypoint_row.addWidget(self.keypoint_type_label)
        keypoint_row.addWidget(self.keypoint_type_combo)
        right_panel.addLayout(keypoint_row)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.frame_slider)
        slider_row.addWidget(self.frame_label)
        right_panel.addLayout(slider_row)

        button_row = QHBoxLayout()
        button_row.addWidget(self.undo_button)
        button_row.addWidget(self.clear_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.point_label)
        right_panel.addLayout(button_row)

        root = QHBoxLayout()
        root.addLayout(left_panel, stretch=1)
        root.addLayout(right_panel, stretch=4)
        self.setLayout(root)

    def _set_keypoint_type(self, keypoint_type: str):
        self.current_keypoint_type = keypoint_type
        self.canvas.set_active_keypoint_type(keypoint_type)

        if self.keypoint_type_combo.currentText() != keypoint_type:
            self.keypoint_type_combo.blockSignals(True)
            self.keypoint_type_combo.setCurrentText(keypoint_type)
            self.keypoint_type_combo.blockSignals(False)

    def _unannotated_sequences(self):
        return [
            seq
            for seq in self.all_sequences
            if str(seq.resolve()) not in self.annotated_sequences
        ]

    def _sequence_counts_by_subphase(self):
        counts = {}
        for seq in self._unannotated_sequences():
            subphase = get_subphase(seq)
            counts[subphase] = counts.get(subphase, 0) + 1
        return counts

    def _populate_filter(self):
        previous_filter = self.sequence_filter.currentData()
        counts = self._sequence_counts_by_subphase()

        self.sequence_filter.blockSignals(True)
        self.sequence_filter.clear()
        self.sequence_filter.addItem(f"All ({sum(counts.values())})", "All")
        for subphase in sorted(counts):
            self.sequence_filter.addItem(f"{subphase} ({counts[subphase]})", subphase)

        idx = self.sequence_filter.findData(previous_filter)
        if idx >= 0:
            self.sequence_filter.setCurrentIndex(idx)
        else:
            self.sequence_filter.setCurrentIndex(0)

        self.sequence_filter.blockSignals(False)

    def _refresh_sequence_list(self):
        selected_filter = self.sequence_filter.currentData()
        sequences = self._unannotated_sequences()
        if selected_filter and selected_filter != "All":
            sequences = [seq for seq in sequences if get_subphase(seq) == selected_filter]

        self.sequence_list.blockSignals(True)
        self.sequence_list.clear()
        for seq in sequences:
            rel = seq.relative_to(self.dataset_root)
            self.sequence_list.addItem(str(rel))
        self.sequence_list.blockSignals(False)
        self.visible_sequences = sequences

    def _on_sequence_selected(self, row: int):
        if row < 0 or row >= len(getattr(self, "visible_sequences", [])):
            return
        self.load_sequence(self.visible_sequences[row])

    def load_sequence(self, left_img_dir: Path):
        images = [
            p
            for p in left_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images.sort(key=natural_key)

        if not images:
            QMessageBox.warning(self, "No images", f"No images found in:\n{left_img_dir}")
            return

        self.current_sequence = left_img_dir
        self.current_images = images
        self.current_frame_idx = 0
        self.current_keypoints = {
            "affordance": None,
            "tool_tip": None,
        }
        self._set_keypoint_type("affordance")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(images) - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.sequence_label.setText(str(left_img_dir.relative_to(self.dataset_root)))
        self.canvas.clear_keypoint()
        self._update_point_label()
        self._show_frame(0)

    def _show_frame(self, idx: int):
        if not self.current_images:
            return

        idx = max(0, min(idx, len(self.current_images) - 1))
        self.current_frame_idx = idx
        image_path = self.current_images[idx]
        self.canvas.set_image(image_path)
        self.canvas.set_keypoints(self.current_keypoints)
        self.frame_label.setText(f"Frame: {idx} / {len(self.current_images) - 1}")

    def _on_slider_changed(self, value: int):
        self._show_frame(value)

    def _update_point_label(self):
        def fmt(pt):
            if pt is None:
                return "None"
            return f"[{pt[0]:.1f}, {pt[1]:.1f}]"

        self.point_label.setText(
            f"Affordance: {fmt(self.current_keypoints['affordance'])} | "
            f"Tool tip: {fmt(self.current_keypoints['tool_tip'])}"
        )

    def _on_keypoint_type_changed(self, keypoint_type: str):
        self.current_keypoint_type = keypoint_type
        self.canvas.set_active_keypoint_type(keypoint_type)
        self._update_point_label()

    def _on_point_changed(self, keypoint_type: str, point_xy):
        self.current_keypoints[keypoint_type] = point_xy
        self._update_point_label()

        # After placing affordance, automatically switch to tool_tip.
        if keypoint_type == "affordance":
            self._set_keypoint_type("tool_tip")

        # After placing tool_tip, automatically save the annotation.
        elif keypoint_type == "tool_tip":
            self._save_annotation()

    def _clear_keypoint(self):
        self.current_keypoints[self.current_keypoint_type] = None
        self.canvas.clear_keypoint(self.current_keypoint_type)
        self._update_point_label()

    def _rewrite_annotated_sequences_file(self):
        annotated_path = self.output_dir / "annotated_sequences.txt"
        tmp_path = annotated_path.with_suffix(".txt.tmp")
        with tmp_path.open("w") as f:
            for seq_str in sorted(self.annotated_sequences):
                f.write(seq_str + "\n")
        tmp_path.replace(annotated_path)

    def _delete_saved_annotation(self, left_img_dir: Path):
        out_name = safe_annotation_name(left_img_dir, self.dataset_root)
        out_path = self.output_dir / constants.ANNOTATIONS_SUBDIR_NAME / out_name

        if out_path.exists():
            out_path.unlink()

        self.annotated_sequences.discard(str(left_img_dir.resolve()))
        self._rewrite_annotated_sequences_file()

    def _save_annotation(self):
        if self.current_sequence is None:
            QMessageBox.warning(self, "No sequence", "No sequence is currently loaded.")
            return

        if self.current_keypoints["affordance"] is None or self.current_keypoints["tool_tip"] is None:
            QMessageBox.warning(self, "Missing keypoints", "Please click both keypoints before saving.")
            return

        image_path = self.current_images[self.current_frame_idx]
        image = QImage(str(image_path))
        if image.isNull():
            QMessageBox.warning(self, "Image error", f"Could not read image:\n{image_path}")
            return

        sequence_to_mark = self.current_sequence

        annotation = {
            "image_path": str(image_path.resolve()),
            "frame_idx": int(self.current_frame_idx),
            "image_size_wh": [int(image.width()), int(image.height())],
            "affordance_xy": [
                float(self.current_keypoints["affordance"][0]),
                float(self.current_keypoints["affordance"][1]),
            ],
            "tool_tip_xy": [
                float(self.current_keypoints["tool_tip"][0]),
                float(self.current_keypoints["tool_tip"][1]),
            ],
        }

        self.writer.save(sequence_to_mark, annotation)
        self.last_saved_sequence = sequence_to_mark
        self.undo_button.setEnabled(True)

        self.annotated_sequences.add(str(sequence_to_mark.resolve()))
        current_row = self.sequence_list.currentRow()

        self._populate_filter()
        self._refresh_sequence_list()

        if self.sequence_list.count() > 0:
            next_row = min(current_row, self.sequence_list.count() - 1)
            self.sequence_list.setCurrentRow(next_row)
        else:
            self.current_sequence = None
            self.current_images = []
            self.canvas.clear_keypoint()
            self.sequence_label.setText("No remaining sequences for this filter")
            self.frame_label.setText("Frame: - / -")
            self.current_keypoints = {
                "affordance": None,
                "tool_tip": None,
            }
            self._set_keypoint_type("affordance")
            self._update_point_label()

    def _select_sequence_in_list(self, left_img_dir: Path):
        target = left_img_dir.resolve()
        for row, seq in enumerate(getattr(self, "visible_sequences", [])):
            if seq.resolve() == target:
                self.sequence_list.setCurrentRow(row)
                return True
        return False

    def _undo_last_save(self):
        if self.last_saved_sequence is None:
            QMessageBox.information(self, "Nothing to undo", "There is no saved annotation to undo.")
            return

        # Make sure any queued save has finished before undoing.
        self.writer.flush()

        sequence_to_restore = self.last_saved_sequence
        self._delete_saved_annotation(sequence_to_restore)

        self.last_saved_sequence = None
        self.undo_button.setEnabled(False)

        self._populate_filter()
        self._refresh_sequence_list()

        restored = self._select_sequence_in_list(sequence_to_restore)
        if not restored:
            self.current_sequence = None
            self.current_images = []
            self.canvas.clear_keypoint()
            self.sequence_label.setText("No sequence loaded")
            self.frame_label.setText("Frame: - / -")
            self.current_keypoints = {
                "affordance": None,
                "tool_tip": None,
            }
            self._set_keypoint_type("affordance")
            self._update_point_label()

    def _rescan_dataset(self):
        self.writer.flush()
        self.all_sequences = load_or_create_sequence_index(
            self.dataset_root,
            self.output_dir,
            force_rescan=True,
        )
        self.annotated_sequences = load_annotated_sequences(self.output_dir)
        self.last_saved_sequence = None
        self.undo_button.setEnabled(False)
        self._populate_filter()
        self._refresh_sequence_list()
        QMessageBox.information(self, "Rescan complete", "Dataset index has been refreshed.")


def main():
    from srth_new.general.constants import RAW_DATASET_ROOT, RAW_ANNOTATION_OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rescan", action="store_true")
    args = parser.parse_args()

    app = QApplication([])
    gui = AnnotatorGUI(
        dataset_root=RAW_DATASET_ROOT,
        output_dir=RAW_ANNOTATION_OUTPUT_DIR,
        force_rescan=args.force_rescan,
    )
    gui.resize(1500, 900)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()