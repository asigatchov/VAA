#!/usr/bin/env python3
"""Build a YOLO dataset from a VAA project JSON using grayscale prev/current/next frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from tqdm import tqdm

BALL_DIAMETER_PX = 20
BALL_RADIUS_PX = BALL_DIAMETER_PX // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create YOLO images and labels from a VAA project. "
            "Each output image is a 3-channel grayscale superframe built from "
            "frame-1, frame, frame+1, while labels come from the current frame only."
        )
    )
    parser.add_argument("--project_path", required=True, help="Path to VAA project JSON")
    parser.add_argument("--data_dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    parser.add_argument(
        "--ball",
        action="store_true",
        help=(
            "Burn the tracked ball into each source frame before building the "
            "grayscale superframe using a 20px diameter marker."
        ),
    )
    return parser.parse_args()


def read_project(project_path: Path) -> dict:
    with open(project_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_ball_data(ball_data) -> list[list[int]]:
    normalized: list[list[int]] = []
    if not isinstance(ball_data, list):
        return normalized

    for row in ball_data:
        if isinstance(row, dict):
            try:
                normalized.append([
                    int(row.get("Frame", -1)),
                    int(row.get("Visibility", 0)),
                    int(float(row.get("X", -1))),
                    int(float(row.get("Y", -1))),
                ])
            except (TypeError, ValueError):
                continue
        elif isinstance(row, (list, tuple)) and len(row) >= 4:
            try:
                normalized.append([
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                ])
            except (TypeError, ValueError):
                continue

    return sorted(normalized, key=lambda row: row[0])


def load_ball_data(project: dict) -> dict[int, tuple[int, int]]:
    ball_rows = normalize_ball_data(project.get("ball_data", []))
    if not ball_rows:
        raw_ball_csv_path = project.get("ball_csv_path") or project.get("description", {}).get("ball_csv_path")
        if raw_ball_csv_path:
            ball_csv_path = Path(raw_ball_csv_path).expanduser()
            if ball_csv_path.exists():
                with open(ball_csv_path, "r", encoding="utf-8", newline="") as file_obj:
                    reader = csv.DictReader(file_obj)
                    ball_rows = normalize_ball_data(list(reader))

    ball_positions: dict[int, tuple[int, int]] = {}
    for frame_idx, visibility, x, y in ball_rows:
        if visibility != 1 or x < 0 or y < 0:
            continue
        ball_positions[frame_idx] = (x, y)

    return ball_positions


def read_frame(cap: cv2.VideoCapture, frame_idx: int, total_frames: int):
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def burn_ball(gray_frame, ball_position: tuple[int, int] | None):
    if ball_position is None:
        return gray_frame

    cv2.circle(gray_frame, ball_position, BALL_RADIUS_PX, 255, thickness=-1, lineType=cv2.LINE_AA)
    return gray_frame


def build_grayscale_superframe(
    gray_frames: dict[int, object],
    frame_idx: int,
    total_frames: int,
    ball_positions: dict[int, tuple[int, int]] | None = None,
):
    indices = [frame_idx - 1, frame_idx, frame_idx + 1]
    merged_frames = []
    for idx in indices:
        clamped_idx = max(0, min(idx, total_frames - 1))
        gray_frame = gray_frames[clamped_idx].copy()
        if ball_positions is not None:
            burn_ball(gray_frame, ball_positions.get(clamped_idx))
        merged_frames.append(gray_frame)
    return cv2.merge(merged_frames)


def collect_required_frame_indices(export_plan: list[tuple[int, dict]], total_frames: int) -> list[int]:
    required_indices: set[int] = set()
    for frame_idx, _ in export_plan:
        for idx in (frame_idx - 1, frame_idx, frame_idx + 1):
            required_indices.add(max(0, min(idx, total_frames - 1)))
    return sorted(required_indices)


def extract_gray_frames(
    cap: cv2.VideoCapture,
    required_indices: list[int],
    total_frames: int,
    disable_progress: bool,
) -> dict[int, object]:
    gray_frames: dict[int, object] = {}
    required_index_set = set(required_indices)
    if not required_index_set:
        return gray_frames

    progress = tqdm(
        total=total_frames,
        desc="Reading source frames",
        unit="frame",
        disable=disable_progress,
    )
    try:
        frame_idx = 0
        while frame_idx < total_frames:
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"Failed during sequential read at frame {frame_idx}")
            if frame_idx in required_index_set:
                gray_frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(gray_frames) == len(required_index_set):
                    progress.update(1)
                    break
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()

    missing_indices = sorted(required_index_set.difference(gray_frames))
    if missing_indices:
        raise RuntimeError(
            f"Failed to extract required frames: {missing_indices[:10]}"
            f"{' ...' if len(missing_indices) > 10 else ''}"
        )
    return gray_frames


def write_yolo_label(label_path: Path, frame_boxes: dict) -> None:
    with open(label_path, "w", encoding="utf-8") as file_obj:
        for _, box_data in sorted(frame_boxes.items(), key=lambda item: int(item[0])):
            class_id, x_center, y_center, width, height = box_data
            file_obj.write(
                f"{int(class_id)} {float(x_center):.10f} {float(y_center):.10f} "
                f"{float(width):.10f} {float(height):.10f}\n"
            )


def group_consecutive_frames(frame_indices: list[int]) -> list[tuple[int, int]]:
    if not frame_indices:
        return []

    groups: list[tuple[int, int]] = []
    start = frame_indices[0]
    end = frame_indices[0]

    for frame_idx in frame_indices[1:]:
        if frame_idx == end + 1:
            end = frame_idx
            continue
        groups.append((start, end))
        start = frame_idx
        end = frame_idx

    groups.append((start, end))
    return groups


def build_export_plan(yolo_boxes: dict, total_frames: int) -> list[tuple[int, dict]]:
    annotated_frame_map = {
        int(frame_key): frame_boxes for frame_key, frame_boxes in yolo_boxes.items() if frame_boxes
    }
    annotated_frames = sorted(annotated_frame_map)
    if not annotated_frames:
        return []

    export_frames: dict[int, dict] = dict(annotated_frame_map)
    annotated_frame_set = set(annotated_frames)

    for start, end in group_consecutive_frames(annotated_frames):
        for offset in (2, 1):
            before_frame_idx = start - offset
            if 0 <= before_frame_idx < total_frames and before_frame_idx not in annotated_frame_set:
                export_frames.setdefault(before_frame_idx, {})

        for offset in (1, 2):
            after_frame_idx = end + offset
            if 0 <= after_frame_idx < total_frames and after_frame_idx not in annotated_frame_set:
                export_frames.setdefault(after_frame_idx, {})

        before_frame_idx = start - 7
        if 0 <= before_frame_idx < total_frames and before_frame_idx not in annotated_frame_set:
            export_frames.setdefault(before_frame_idx, {})

        after_frame_idx = end + 7
        if 0 <= after_frame_idx < total_frames and after_frame_idx not in annotated_frame_set:
            export_frames.setdefault(after_frame_idx, {})

    return sorted(export_frames.items(), key=lambda item: item[0])


def main() -> int:
    args = parse_args()
    project_path = Path(args.project_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if not project_path.exists():
        raise FileNotFoundError(f"Project JSON not found: {project_path}")

    project = read_project(project_path)
    video_path = Path(project["video_path"]).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    yolo_boxes = project.get("annotations", {}).get("yolo_boxes", {})
    if not yolo_boxes:
        raise ValueError("Project has no annotations.yolo_boxes to export")
    ball_positions = load_ball_data(project) if args.ball else None

    project_name = project.get("name") or project_path.stem
    output_root = data_dir / project_name
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    exported_count = 0
    empty_count = 0
    export_plan = build_export_plan(yolo_boxes, total_frames)
    if not export_plan:
        raise ValueError("Project has no non-empty annotations.yolo_boxes to export")
    required_frame_indices = collect_required_frame_indices(export_plan, total_frames)

    try:
        gray_frames = extract_gray_frames(
            capture,
            required_frame_indices,
            total_frames,
            disable_progress=args.no_progress,
        )
        progress = tqdm(
            export_plan,
            desc="Exporting YOLO samples",
            unit="frame",
            disable=args.no_progress,
        )
        for frame_idx, frame_boxes in progress:
            image = build_grayscale_superframe(
                gray_frames,
                frame_idx,
                total_frames,
                ball_positions=ball_positions,
            )
            stem = f"{project_name}_{frame_idx:06d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to write image: {image_path}")
            write_yolo_label(label_path, frame_boxes)
            exported_count += 1
            if not frame_boxes:
                empty_count += 1
    finally:
        capture.release()

    summary = {
        "project_path": str(project_path),
        "video_path": str(video_path),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "num_samples": exported_count,
        "num_empty_context_samples": empty_count,
        "num_annotated_samples": exported_count - empty_count,
        "ball_burned": bool(args.ball),
        "ball_marker_diameter_px": BALL_DIAMETER_PX if args.ball else 0,
        "num_ball_positions": len(ball_positions) if ball_positions is not None else 0,
    }
    with open(output_root / "dataset_manifest.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
