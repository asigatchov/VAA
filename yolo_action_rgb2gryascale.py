#!/usr/bin/env python3
"""Build a YOLO dataset from a VAA project JSON using grayscale prev/current/next frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


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
    return parser.parse_args()


def read_project(project_path: Path) -> dict:
    with open(project_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_frame(cap: cv2.VideoCapture, frame_idx: int, total_frames: int):
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def build_grayscale_superframe(cap: cv2.VideoCapture, frame_idx: int, total_frames: int):
    indices = [frame_idx - 1, frame_idx, frame_idx + 1]
    gray_frames = []
    for idx in indices:
        frame = read_frame(cap, idx, total_frames)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return cv2.merge(gray_frames)


def write_yolo_label(label_path: Path, frame_boxes: dict) -> None:
    with open(label_path, "w", encoding="utf-8") as file_obj:
        for _, box_data in sorted(frame_boxes.items(), key=lambda item: int(item[0])):
            class_id, x_center, y_center, width, height = box_data
            file_obj.write(
                f"{int(class_id)} {float(x_center):.10f} {float(y_center):.10f} "
                f"{float(width):.10f} {float(height):.10f}\n"
            )


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

    try:
        for frame_key in sorted(yolo_boxes, key=lambda value: int(value)):
            frame_idx = int(frame_key)
            frame_boxes = yolo_boxes[frame_key]
            if not frame_boxes:
                continue

            image = build_grayscale_superframe(capture, frame_idx, total_frames)
            stem = f"{project_name}_{frame_idx:06d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to write image: {image_path}")
            write_yolo_label(label_path, frame_boxes)
            exported_count += 1
    finally:
        capture.release()

    summary = {
        "project_path": str(project_path),
        "video_path": str(video_path),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "num_samples": exported_count,
    }
    with open(output_root / "dataset_manifest.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
