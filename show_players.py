from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rfdetr
from rfdetr.assets.coco_classes import COCO_CLASSES


PERSON_CLASS_NAME = "person"
BALL_CLASS_NAME = "sports ball"
MODEL_VARIANTS = {
    "nano": "RFDETRNano",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
}


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show player and ball predictions from RF-DETR using cv2."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a video file or project JSON.",
    )
    parser.add_argument(
        "--model-variant",
        choices=tuple(MODEL_VARIANTS.keys()),
        default="base",
        help="RF-DETR model variant.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="RF-DETR input resolution.",
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold for person detections.",
    )
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=0.12,
        help="Confidence threshold for ball detections.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Show every Nth frame.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show side-by-side comparison: raw RF-DETR boxes vs UI roundtrip boxes.",
    )
    return parser.parse_args()


def resolve_video_path(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        video_path = Path(payload.get("video_path") or payload.get("description", {}).get("video_path", "")).expanduser()
        if not video_path.is_absolute():
            video_path = (input_path.parent / video_path).resolve()
        return video_path
    return input_path


def build_model(variant: str, resolution: int):
    class_name = MODEL_VARIANTS[variant]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        supported = ", ".join(sorted(name for name in MODEL_VARIANTS if getattr(rfdetr, MODEL_VARIANTS[name], None)))
        raise SystemExit(
            f"RF-DETR variant '{variant}' is unavailable in the installed rfdetr package. "
            f"Available variants: {supported or 'none'}"
        )
    model = model_cls(resolution=resolution)
    model.optimize_for_inference()
    return model


def run_model(model, frame_bgr: np.ndarray, threshold: float) -> list[Detection]:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    detections = model.predict(rgb, threshold=threshold)
    if detections.xyxy is None:
        return []

    parsed: list[Detection] = []
    for index in range(len(detections.xyxy)):
        parsed.append(
            Detection(
                class_name=COCO_CLASSES[int(detections.class_id[index])],
                confidence=float(detections.confidence[index]),
                xyxy=tuple(float(value) for value in detections.xyxy[index].tolist()),
            )
        )
    return parsed


def draw_detection(frame: np.ndarray, detection: Detection, color: tuple[int, int, int], label_prefix: str) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{label_prefix} {detection.confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - text_height - baseline - 4)
    cv2.rectangle(frame, (x1, top), (x1 + text_width + 8, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(text_height + 2, y1 - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def xyxy_to_normalized(
    xyxy: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    frame_width, frame_height = frame_size
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(frame_width), x1))
    y1 = max(0.0, min(float(frame_height), y1))
    x2 = max(0.0, min(float(frame_width), x2))
    y2 = max(0.0, min(float(frame_height), y2))
    return (
        ((x1 + x2) / 2.0) / frame_width,
        ((y1 + y2) / 2.0) / frame_height,
        (x2 - x1) / frame_width,
        (y2 - y1) / frame_height,
    )


def normalized_to_xyxy(
    box: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = box
    frame_width, frame_height = frame_size
    x1 = int(round((x_center - width / 2.0) * frame_width))
    y1 = int(round((y_center - height / 2.0) * frame_height))
    x2 = int(round((x_center + width / 2.0) * frame_width))
    y2 = int(round((y_center + height / 2.0) * frame_height))
    return (
        max(0, min(frame_width, x1)),
        max(0, min(frame_height, y1)),
        max(0, min(frame_width, x2)),
        max(0, min(frame_height, y2)),
    )


def draw_xyxy(frame: np.ndarray, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - text_height - baseline - 4)
    cv2.rectangle(frame, (x1, top), (x1 + text_width + 8, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(text_height + 2, y1 - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def build_debug_view(
    frame: np.ndarray,
    people: list[Detection],
    balls: list[Detection],
    frame_idx: int,
    model_variant: str,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width, frame_height)

    raw_view = frame.copy()
    ui_view = frame.copy()

    for detection in people:
        draw_detection(raw_view, detection, (80, 220, 80), "raw player")
        normalized = xyxy_to_normalized(detection.xyxy, frame_size)
        reproj_xyxy = normalized_to_xyxy(normalized, frame_size)
        draw_xyxy(ui_view, reproj_xyxy, (255, 120, 0), "ui player")

    for detection in balls:
        draw_detection(raw_view, detection, (0, 215, 255), "raw ball")
        normalized = xyxy_to_normalized(detection.xyxy, frame_size)
        reproj_xyxy = normalized_to_xyxy(normalized, frame_size)
        draw_xyxy(ui_view, reproj_xyxy, (255, 0, 255), "ui ball")

    cv2.putText(raw_view, "RAW RF-DETR xyxy", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ui_view, "UI roundtrip", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        raw_view,
        f"frame={frame_idx} model={model_variant}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        ui_view,
        f"frame={frame_idx} model={model_variant}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.hstack([raw_view, ui_view])


def main() -> None:
    args = parse_args()
    video_path = resolve_video_path(args.input_path)
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    model = build_model(args.model_variant, args.resolution)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_step = max(1, int(args.frame_step))
    paused = False
    frame_idx = 0
    window_name = f"show_players | {video_path.name} | RF-DETR {args.model_variant}"

    print("Controls: q=quit, space=pause/resume, n=next frame when paused")

    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_idx = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_step > 1:
                next_target = frame_idx + frame_step - 1
                capture.set(cv2.CAP_PROP_POS_FRAMES, next_target + 1)
                frame_idx = next_target

            detections = run_model(model, frame, threshold=min(args.person_threshold, args.ball_threshold))
            people = [
                detection for detection in detections
                if detection.class_name == PERSON_CLASS_NAME and detection.confidence >= args.person_threshold
            ]
            balls = [
                detection for detection in detections
                if detection.class_name == BALL_CLASS_NAME and detection.confidence >= args.ball_threshold
            ]

            if args.debug:
                display = build_debug_view(frame, people, balls, frame_idx, args.model_variant)
            else:
                display = frame.copy()
                for detection in people:
                    draw_detection(display, detection, (80, 220, 80), "player")
                for detection in balls:
                    draw_detection(display, detection, (0, 215, 255), "ball")

                cv2.putText(
                    display,
                    f"frame={frame_idx} players={len(people)} balls={len(balls)} model={args.model_variant}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, display)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("n") and paused:
            paused = False
            continue

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
