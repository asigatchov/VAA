"""Dataset preparation and loading for the volleyball action detector."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ACTION_CLASS_NAMES, EXPORT_CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FRAME_OFFSETS = tuple(range(-4, 5))
DEFAULT_ACTION_SUPERFRAME_RADIUS = 1
BALL_DIAMETER_PX = 20
ACTION_CLASS_IDS = {0, 1, 2, 3}

CLASS_NAME_ALIASES = {
    "serve": "serve",
    "server": "serve",
    "receive": "receive",
    "recive": "receive",
    "reception": "receive",
    "set": "set",
    "attack": "attack",
    "attak": "attack",
    "player": "player",
    "playery": "player",
    "person": "player",
    "ball": "ball",
    "sports ball": "ball",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in EXPORT_CLASS_NAMES.items()}


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    label_path: str
    source_project: str | None = None
    source_frame: int | None = None


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise ModuleNotFoundError(
            "OpenCV is required for dataset preparation and video inference. "
            "Install project dependencies in the active environment first."
        ) from exc
    return cv2


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("NumPy is required for action detector data loading.") from exc
    return np


def parse_frame_offsets(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_FRAME_OFFSETS
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise ValueError("frame offsets cannot be empty")
    return offsets


def parse_splits(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError("--splits must contain train,valid,test ratios")
    total = sum(parts)
    if total <= 0:
        raise ValueError("--splits sum must be positive")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def normalize_class_name(class_name: str | None) -> str | None:
    if class_name is None:
        return None
    return CLASS_NAME_ALIASES.get(str(class_name).strip().lower())


def remap_class_id(raw_class_id: int, box_classes: dict[int, str]) -> int | None:
    """Map project-specific ids to ``0..5``.

    VAA project JSON may use helper ids ``5=player`` and ``6=ball`` while the
    training dataset uses ``4=player`` and ``5=ball``.
    """

    class_name = normalize_class_name(box_classes.get(raw_class_id))
    if class_name is not None:
        return CLASS_NAME_TO_ID.get(class_name)
    if 0 <= raw_class_id <= 3:
        return raw_class_id
    if raw_class_id == 4:
        return 4
    if raw_class_id == 5:
        return 5
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_ball_rows(ball_data: Any) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    if not isinstance(ball_data, list):
        return rows
    for row in ball_data:
        try:
            if isinstance(row, dict):
                frame = int(float(row.get("Frame", -1)))
                visibility = int(float(row.get("Visibility", 0)))
                x = int(float(row.get("X", -1)))
                y = int(float(row.get("Y", -1)))
            else:
                frame = int(float(row[0]))
                visibility = int(float(row[1]))
                x = int(float(row[2]))
                y = int(float(row[3]))
        except (IndexError, TypeError, ValueError):
            continue
        rows.append((frame, visibility, x, y))
    return sorted(rows)


def load_ball_positions(project: dict[str, Any]) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    for frame_idx, visibility, x, y in normalize_ball_rows(project.get("ball_data", [])):
        if visibility > 0 and x >= 0 and y >= 0:
            positions[frame_idx] = (x, y)
    return positions


def remap_project_boxes(
    frame_boxes: dict[str, list[float]],
    box_classes: dict[int, str],
    frame_width: int,
    frame_height: int,
    ball_position: tuple[int, int] | None = None,
    add_ball_from_tracking: bool = True,
    include_class_ids: set[int] | None = None,
) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    has_ball_label = False
    for _, raw_box in sorted(frame_boxes.items(), key=lambda item: int(item[0])):
        if len(raw_box) < 5:
            continue
        mapped_class_id = remap_class_id(int(raw_box[0]), box_classes)
        if mapped_class_id is None:
            continue
        if include_class_ids is not None and mapped_class_id not in include_class_ids:
            continue
        _, cx, cy, width, height = raw_box[:5]
        width = clamp01(width)
        height = clamp01(height)
        if width <= 0.0 or height <= 0.0:
            continue
        labels.append((mapped_class_id, clamp01(cx), clamp01(cy), width, height))
        has_ball_label = has_ball_label or mapped_class_id == 5

    if (
        add_ball_from_tracking
        and ball_position is not None
        and not has_ball_label
        and (include_class_ids is None or 5 in include_class_ids)
    ):
        x, y = ball_position
        labels.append(
            (
                5,
                clamp01(x / max(1, frame_width)),
                clamp01(y / max(1, frame_height)),
                clamp01(BALL_DIAMETER_PX / max(1, frame_width)),
                clamp01(BALL_DIAMETER_PX / max(1, frame_height)),
            )
        )
    return labels


def group_consecutive(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    groups: list[tuple[int, int]] = []
    start = end = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx == end + 1:
            end = frame_idx
            continue
        groups.append((start, end))
        start = end = frame_idx
    groups.append((start, end))
    return groups


def frame_action_class_ids(frame_boxes: dict, box_classes: dict[int, str]) -> set[int]:
    class_ids: set[int] = set()
    for box in frame_boxes.values():
        if len(box) < 1:
            continue
        mapped_class_id = remap_class_id(int(box[0]), box_classes)
        if mapped_class_id in ACTION_CLASS_IDS:
            class_ids.add(mapped_class_id)
    return class_ids


def build_action_mask_sources(
    yolo_boxes: dict[str, dict],
    box_classes: dict[int, str],
    total_frames: int,
    superframe_radius: int = DEFAULT_ACTION_SUPERFRAME_RADIUS,
) -> dict[int, dict[int, set[int]]]:
    """Map action class -> real video frame -> source annotated center frames.

    A project action label is drawn while viewing a 3-frame superframe, so one
    annotated center frame represents action presence on center-1, center,
    center+1.
    """

    sources: dict[int, dict[int, set[int]]] = {class_id: {} for class_id in ACTION_CLASS_IDS}
    for frame_key, frame_boxes in yolo_boxes.items():
        if not frame_boxes:
            continue
        center_frame = int(frame_key)
        for class_id in frame_action_class_ids(frame_boxes, box_classes):
            for offset in range(-superframe_radius, superframe_radius + 1):
                frame_idx = center_frame + offset
                if 0 <= frame_idx < total_frames:
                    sources[class_id].setdefault(frame_idx, set()).add(center_frame)
    return sources


def merged_action_mask_frames(action_mask_sources: dict[int, dict[int, set[int]]]) -> list[int]:
    frames: set[int] = set()
    for frame_sources in action_mask_sources.values():
        frames.update(frame_sources)
    return sorted(frames)


def build_stack_action_mask(
    center_frame: int,
    offsets: tuple[int, ...],
    total_frames: int,
    action_mask_sources: dict[int, dict[int, set[int]]],
) -> list[int]:
    action_frames = set(merged_action_mask_frames(action_mask_sources))
    return [
        1 if max(0, min(total_frames - 1, center_frame + offset)) in action_frames else 0
        for offset in offsets
    ]


def action_overlap_threshold(offsets: tuple[int, ...]) -> int:
    return (len(offsets) // 2) + 1


def build_export_frames(
    yolo_boxes: dict[str, dict],
    total_frames: int,
    box_classes: dict[int, str] | None = None,
    frame_offsets: tuple[int, ...] = DEFAULT_FRAME_OFFSETS,
) -> list[int]:
    box_classes = box_classes or {}
    annotated = {int(key): value for key, value in yolo_boxes.items() if value}
    export_frames = set(annotated)
    action_mask_sources = build_action_mask_sources(yolo_boxes, box_classes, total_frames)
    action_frames = merged_action_mask_frames(action_mask_sources)
    if action_frames:
        min_offset = min(frame_offsets)
        max_offset = max(frame_offsets)
        for start, end in group_consecutive(action_frames):
            for frame_idx in range(start - max_offset - 1, end - min_offset + 2):
                if 0 <= frame_idx < total_frames:
                    export_frames.add(frame_idx)
    return sorted(export_frames)


def select_temporal_action_sources(
    center_frame: int,
    offsets: tuple[int, ...],
    total_frames: int,
    action_mask_sources: dict[int, dict[int, set[int]]],
    min_overlap: int,
) -> dict[int, int]:
    selected: dict[int, int] = {}
    stack_indices = [max(0, min(total_frames - 1, center_frame + offset)) for offset in offsets]
    for class_id, frame_sources in action_mask_sources.items():
        overlapping_frames = [frame_idx for frame_idx in stack_indices if frame_idx in frame_sources]
        if len(overlapping_frames) < min_overlap:
            continue
        candidate_sources = {
            source_frame
            for frame_idx in overlapping_frames
            for source_frame in frame_sources.get(frame_idx, set())
        }
        if not candidate_sources:
            continue
        selected[class_id] = min(candidate_sources, key=lambda source_frame: (abs(source_frame - center_frame), source_frame))
    return selected


def build_temporal_labels(
    yolo_boxes: dict[str, dict],
    box_classes: dict[int, str],
    frame_idx: int,
    frame_width: int,
    frame_height: int,
    ball_positions: dict[int, tuple[int, int]],
    action_mask_sources: dict[int, dict[int, set[int]]],
    frame_offsets: tuple[int, ...],
    total_frames: int,
    add_ball_from_tracking: bool = True,
    include_class_ids: set[int] | None = None,
) -> tuple[list[tuple[int, float, float, float, float]], list[int], dict[int, int]]:
    min_overlap = action_overlap_threshold(frame_offsets)
    selected_sources = select_temporal_action_sources(
        frame_idx,
        frame_offsets,
        total_frames,
        action_mask_sources,
        min_overlap,
    )
    labels: list[tuple[int, float, float, float, float]] = []
    include_non_action_ids = None
    if include_class_ids is not None:
        include_non_action_ids = include_class_ids - ACTION_CLASS_IDS

    if include_non_action_ids is None or include_non_action_ids:
        labels.extend(
            remap_project_boxes(
                yolo_boxes.get(str(frame_idx), {}),
                box_classes,
                frame_width,
                frame_height,
                ball_positions.get(frame_idx),
                add_ball_from_tracking=add_ball_from_tracking,
                include_class_ids=include_non_action_ids,
            )
        )

    for class_id, source_frame in selected_sources.items():
        if include_class_ids is not None and class_id not in include_class_ids:
            continue
        labels.extend(
            remap_project_boxes(
                yolo_boxes.get(str(source_frame), {}),
                box_classes,
                frame_width,
                frame_height,
                None,
                add_ball_from_tracking=False,
                include_class_ids={class_id},
            )
        )

    if selected_sources and (include_non_action_ids is None or include_non_action_ids):
        has_player_label = any(int(label[0]) == 4 for label in labels)
        nearest_source = min(set(selected_sources.values()), key=lambda source_frame: (abs(source_frame - frame_idx), source_frame))
        source_helper_ids = set(include_non_action_ids or {4, 5})
        if not has_player_label and 4 in source_helper_ids:
            labels.extend(
                remap_project_boxes(
                    yolo_boxes.get(str(nearest_source), {}),
                    box_classes,
                    frame_width,
                    frame_height,
                    None,
                    add_ball_from_tracking=False,
                    include_class_ids={4},
                )
            )

    action_mask = build_stack_action_mask(frame_idx, frame_offsets, total_frames, action_mask_sources)
    return labels, action_mask, selected_sources


def collect_required_indices(frames: list[int], offsets: tuple[int, ...], total_frames: int) -> list[int]:
    required = {
        max(0, min(total_frames - 1, frame_idx + offset))
        for frame_idx in frames
        for offset in offsets
    }
    return sorted(required)


def extract_gray_frames(video_path: Path, required_indices: list[int]) -> tuple[dict[int, Any], dict[str, int]]:
    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    required = set(required_indices)
    frames: dict[int, Any] = {}
    try:
        frame_idx = 0
        while frame_idx < total_frames and len(frames) < len(required):
            success, frame = capture.read()
            if not success:
                break
            if frame_idx in required:
                frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1
    finally:
        capture.release()
    missing = sorted(required - set(frames))
    if missing:
        raise RuntimeError(f"Failed to read required frames from {video_path}: {missing[:10]}")
    return frames, {"total_frames": total_frames, "width": frame_width, "height": frame_height}


def build_stack(
    gray_frames: dict[int, Any],
    center_frame: int,
    offsets: tuple[int, ...],
    total_frames: int,
    image_size: tuple[int, int],
) -> Any:
    cv2 = require_cv2()
    np = require_numpy()
    out_h, out_w = image_size
    channels = []
    for offset in offsets:
        frame_idx = max(0, min(total_frames - 1, center_frame + offset))
        gray = gray_frames[frame_idx]
        if gray.shape[:2] != (out_h, out_w):
            gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        channels.append(gray)
    return np.stack(channels, axis=0).astype("uint8")


def write_label_file(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for class_id, cx, cy, width, height in labels:
            file_obj.write(f"{class_id} {cx:.10f} {cy:.10f} {width:.10f} {height:.10f}\n")


def split_records(records: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    train_ratio, valid_ratio, _ = ratios
    train_end = int(round(len(shuffled) * train_ratio))
    valid_end = train_end + int(round(len(shuffled) * valid_ratio))
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def discover_project_paths(projects_path: Path) -> list[Path]:
    if projects_path.is_file():
        return [projects_path]
    paths = [
        path
        for path in projects_path.rglob("*.json")
        if path.name != "dataset_manifest.json" and "datasets" not in path.parts
    ]
    return sorted(paths)


def prepare_projects(
    project_paths: list[Path],
    output_dir: Path,
    image_size: tuple[int, int] = (432, 768),
    frame_offsets: tuple[int, ...] = DEFAULT_FRAME_OFFSETS,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    add_ball_from_tracking: bool = True,
    action_only: bool = False,
) -> dict[str, Any]:
    np = require_numpy()
    all_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = ACTION_CLASS_NAMES if action_only else EXPORT_CLASS_NAMES
    include_class_ids = set(class_names)

    for project_path in project_paths:
        project = read_json(project_path)
        video_path = Path(project["video_path"]).expanduser()
        annotations = project.get("annotations", {})
        yolo_boxes = annotations.get("yolo_boxes", {})
        box_classes = {int(key): value for key, value in annotations.get("box_classes", {}).items()}
        ball_positions = load_ball_positions(project)

        if not yolo_boxes:
            continue
        cv2 = require_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        action_mask_sources = build_action_mask_sources(yolo_boxes, box_classes, total_frames)
        export_frames = build_export_frames(yolo_boxes, total_frames, box_classes, frame_offsets)
        required = collect_required_indices(export_frames, frame_offsets, total_frames)
        gray_frames, video_info = extract_gray_frames(video_path, required)
        project_name = str(project.get("name") or project_path.stem)

        for frame_idx in export_frames:
            stack = build_stack(gray_frames, frame_idx, frame_offsets, total_frames, image_size)
            labels, action_mask, selected_action_sources = build_temporal_labels(
                yolo_boxes,
                box_classes,
                frame_idx,
                frame_width,
                frame_height,
                ball_positions,
                action_mask_sources,
                frame_offsets,
                total_frames,
                add_ball_from_tracking=add_ball_from_tracking,
                include_class_ids=include_class_ids,
            )
            stem = f"{project_name}_{frame_idx:06d}"
            image_rel = Path("all") / "images" / f"{stem}.npz"
            label_rel = Path("all") / "labels" / f"{stem}.txt"
            image_path = output_dir / image_rel
            label_path = output_dir / label_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(image_path, image=stack, action_mask=np.asarray(action_mask, dtype="uint8"))
            write_label_file(label_path, labels)
            all_records.append(
                {
                    "image_path": str(image_rel),
                    "label_path": str(label_rel),
                    "source_project": str(project_path),
                    "source_video": str(video_path),
                    "source_frame": frame_idx,
                    "num_labels": len(labels),
                    "action_mask": action_mask,
                    "selected_action_sources": {
                        str(class_id): source_frame for class_id, source_frame in selected_action_sources.items()
                    },
                }
            )

        write_json(
            output_dir / "projects" / f"{project_name}.json",
            {
                "project_path": str(project_path),
                "video_path": str(video_path),
                "video_info": video_info,
                "num_samples": len(export_frames),
            },
        )

    split_map = split_records(all_records, splits, seed)
    for split_name, records in split_map.items():
        split_dir = output_dir / split_name
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        split_records_out = []
        for record in records:
            source_image = output_dir / record["image_path"]
            source_label = output_dir / record["label_path"]
            target_image = split_dir / "images" / source_image.name
            target_label = split_dir / "labels" / source_label.name
            if not target_image.exists():
                target_image.hardlink_to(source_image)
            if not target_label.exists():
                target_label.hardlink_to(source_label)
            split_records_out.append(
                {
                    **record,
                    "image_path": str(Path(split_name) / "images" / source_image.name),
                    "label_path": str(Path(split_name) / "labels" / source_label.name),
                }
            )
        write_json(split_dir / "manifest.json", {"records": split_records_out})

    class_counts = {str(class_id): 0 for class_id in class_names}
    for record in all_records:
        with open(output_dir / record["label_path"], "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                class_id_text = line.split()[0]
                if class_id_text in class_counts:
                    class_counts[class_id_text] += 1

    summary = {
        "format": "vaa-action-stack-v1",
        "num_samples": len(all_records),
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "frame_offsets": list(frame_offsets),
        "in_dim": len(frame_offsets),
        "action_superframe_radius": DEFAULT_ACTION_SUPERFRAME_RADIUS,
        "action_overlap_threshold": action_overlap_threshold(frame_offsets),
        "action_label_rule": "positive when action_mask sum for a class is >= action_overlap_threshold",
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": {name: len(records) for name, records in split_map.items()},
        "class_counts": class_counts,
    }
    write_json(output_dir / "dataset_manifest.json", summary)
    return summary


class ActionDetectionDataset:
    """Dataset for prepared ``.npz`` stacks or YOLO-style image folders."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (432, 768),
        in_dim: int | None = None,
        augment: bool = False,
        hflip_p: float = 0.0,
        rotate_degrees: float = 0.0,
        rotate_p: float = 0.0,
    ):
        torch = self._require_torch()
        self.torch = torch
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.in_dim = in_dim
        self.augment = augment
        self.hflip_p = float(hflip_p)
        self.rotate_degrees = float(rotate_degrees)
        self.rotate_p = float(rotate_p)
        self.records = self._discover_records()
        if not self.records:
            raise ValueError(f"No samples found in {self.dataset_dir} split={split}")

    @staticmethod
    def _require_torch():
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("PyTorch is required for ActionDetectionDataset.") from exc
        return torch

    def _discover_records(self) -> list[SampleRecord]:
        manifest_path = self.dataset_dir / self.split / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            return [
                SampleRecord(
                    image_path=str(self.dataset_dir / record["image_path"]),
                    label_path=str(self.dataset_dir / record["label_path"]),
                    source_project=record.get("source_project"),
                    source_frame=record.get("source_frame"),
                )
                for record in manifest.get("records", [])
            ]

        split_dir = self.dataset_dir / self.split
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if not image_dir.is_dir():
            image_dir = self.dataset_dir / "images"
            label_dir = self.dataset_dir / "labels"
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() == ".npz"
        )
        return [
            SampleRecord(
                image_path=str(path),
                label_path=str(label_dir / f"{path.stem}.txt"),
            )
            for path in image_paths
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path):
        np = require_numpy()
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            image = data["image"]
        else:
            cv2 = require_cv2()
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"Failed to read image: {path}")
            if raw.ndim == 2:
                image = raw[None, :, :]
            else:
                image = raw.transpose(2, 0, 1)
        if image.shape[1:] != self.image_size:
            cv2 = require_cv2()
            resized = [
                cv2.resize(channel, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                for channel in image
            ]
            image = np.stack(resized, axis=0)
        if self.in_dim is not None and image.shape[0] != self.in_dim:
            if image.shape[0] == 1:
                image = np.repeat(image, self.in_dim, axis=0)
            elif image.shape[0] < self.in_dim:
                pad = np.repeat(image[-1:], self.in_dim - image.shape[0], axis=0)
                image = np.concatenate([image, pad], axis=0)
            else:
                center = image.shape[0] // 2
                half = self.in_dim // 2
                start = max(0, center - half)
                image = image[start : start + self.in_dim]
        return self.torch.from_numpy(image.astype("float32") / 255.0)

    def _load_labels(self, path: Path):
        rows: list[list[float]] = []
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    rows.append([float(part) for part in parts])
        if not rows:
            return self.torch.zeros((0, 5), dtype=self.torch.float32)
        return self.torch.tensor(rows, dtype=self.torch.float32)

    def _apply_hflip(self, image, labels):
        image = image[:, :, ::-1].copy()
        if labels.numel() > 0:
            labels = labels.clone()
            labels[:, 1] = 1.0 - labels[:, 1]
        return image, labels

    def _apply_rotation(self, image, labels, degrees: float):
        cv2 = require_cv2()
        np = require_numpy()
        height, width = image.shape[1:]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = np.stack(
            [
                cv2.warpAffine(
                    channel,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for channel in image
            ],
            axis=0,
        )
        if labels.numel() == 0:
            return rotated, labels

        transformed = []
        for row in labels.tolist():
            class_id, cx, cy, box_w, box_h = row
            x1 = (cx - box_w / 2.0) * width
            y1 = (cy - box_h / 2.0) * height
            x2 = (cx + box_w / 2.0) * width
            y2 = (cy + box_h / 2.0) * height
            corners = np.array(
                [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
                dtype=np.float32,
            )
            mapped = corners @ matrix.T
            min_x = float(np.clip(mapped[:, 0].min(), 0, width - 1))
            max_x = float(np.clip(mapped[:, 0].max(), 0, width - 1))
            min_y = float(np.clip(mapped[:, 1].min(), 0, height - 1))
            max_y = float(np.clip(mapped[:, 1].max(), 0, height - 1))
            new_w = max_x - min_x
            new_h = max_y - min_y
            if new_w < 2.0 or new_h < 2.0:
                continue
            transformed.append(
                [
                    class_id,
                    ((min_x + max_x) / 2.0) / width,
                    ((min_y + max_y) / 2.0) / height,
                    new_w / width,
                    new_h / height,
                ]
            )
        if not transformed:
            return rotated, self.torch.zeros((0, 5), dtype=self.torch.float32)
        return rotated, self.torch.tensor(transformed, dtype=self.torch.float32)

    def _augment(self, image, labels):
        if not self.augment:
            return image, labels
        np = require_numpy()
        if self.hflip_p > 0.0 and float(np.random.random()) < self.hflip_p:
            image, labels = self._apply_hflip(image, labels)
        if self.rotate_degrees > 0.0 and self.rotate_p > 0.0 and float(np.random.random()) < self.rotate_p:
            angle = float(np.random.uniform(-self.rotate_degrees, self.rotate_degrees))
            image, labels = self._apply_rotation(image, labels, angle)
        return image, labels

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self._load_image(Path(record.image_path))
        labels = self._load_labels(Path(record.label_path))
        image_np = image.numpy()
        image_np, labels = self._augment(image_np, labels)
        image = self.torch.from_numpy(image_np.copy())
        meta = {
            "image_path": record.image_path,
            "label_path": record.label_path,
            "source_project": record.source_project,
            "source_frame": record.source_frame,
        }
        return image, labels, meta


def collate_detection_batch(batch):
    torch = ActionDetectionDataset._require_torch()
    images, labels, metas = zip(*batch)
    return torch.stack(images, dim=0), list(labels), list(metas)


def parse_prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VAA action-detector training data.")
    parser.add_argument("--project", action="append", default=[], help="Path to a VAA project JSON or directory.")
    parser.add_argument("--projects-path", help="Directory with VAA project JSON files.")
    parser.add_argument("--output-dir", default="data/action_detector", help="Output dataset directory.")
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frame-offsets", default=",".join(str(x) for x in DEFAULT_FRAME_OFFSETS))
    parser.add_argument("--splits", default="0.8,0.1,0.1", help="train,valid,test ratios.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ball-from-tracking", action="store_true")
    parser.add_argument("--action-only", action="store_true", help="Keep only action classes 0..3 in labels.")
    return parser.parse_args()


def main() -> int:
    args = parse_prepare_args()
    project_paths: list[Path] = []
    for project in args.project:
        project_paths.extend(discover_project_paths(Path(project).expanduser()))
    if args.projects_path:
        project_paths.extend(discover_project_paths(Path(args.projects_path).expanduser()))
    project_paths = sorted(set(path.resolve() for path in project_paths))
    if not project_paths:
        raise ValueError("Provide at least one --project or --projects-path")

    summary = prepare_projects(
        project_paths=project_paths,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        image_size=(args.height, args.width),
        frame_offsets=parse_frame_offsets(args.frame_offsets),
        splits=parse_splits(args.splits),
        seed=args.seed,
        add_ball_from_tracking=not args.no_ball_from_tracking,
        action_only=args.action_only,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
