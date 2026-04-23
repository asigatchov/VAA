"""Run VballActionDetector inference on video or prepared stack files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path

import torch

from .action_detector import (
    ACTION_CLASS_NAMES,
    EXPORT_CLASS_NAMES,
    VballActionDetector,
    VballActionDetectorV2,
    postprocess_predictions,
    postprocess_predictions_v2,
)
from .data import parse_frame_offsets, require_cv2, require_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer volleyball actions with VballActionDetector.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/last.pt.")
    parser.add_argument("--video", help="Input video path.")
    parser.add_argument("--stack", action="append", default=[], help="Prepared .npz stack file. Can be repeated.")
    parser.add_argument("--output-json", default="action_detections.json")
    parser.add_argument("--output-video", help="Optional annotated video path.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--frame-offsets", default=None)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame for video input.")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--actions-only", action="store_true", help="Drop helper player/ball detections.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[VballActionDetector | VballActionDetectorV2, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    in_dim = int(config.get("in_dim", 9))
    num_classes = int(config.get("num_classes", 6))
    width_mult = float(config.get("width_mult", 1.0))
    model_variant = str(config.get("model_variant", "v1"))
    if model_variant == "v2":
        model = VballActionDetectorV2(in_dim=in_dim, num_classes=num_classes, width_mult=width_mult)
    else:
        model = VballActionDetector(in_dim=in_dim, num_classes=num_classes, width_mult=width_mult)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config


class FrameCache:
    def __init__(self, capture, limit: int = 64):
        self.capture = capture
        self.limit = limit
        self.cache: OrderedDict[int, object] = OrderedDict()

    def get_gray(self, frame_idx: int):
        cv2 = require_cv2()
        if frame_idx in self.cache:
            value = self.cache.pop(frame_idx)
            self.cache[frame_idx] = value
            return value
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cache[frame_idx] = gray
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return gray

    def get_bgr(self, frame_idx: int):
        cv2 = require_cv2()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return frame


def build_video_stack(
    cache: FrameCache,
    frame_idx: int,
    total_frames: int,
    offsets: tuple[int, ...],
    image_size: tuple[int, int],
):
    cv2 = require_cv2()
    np = require_numpy()
    out_h, out_w = image_size
    channels = []
    for offset in offsets:
        source_idx = max(0, min(total_frames - 1, frame_idx + offset))
        gray = cache.get_gray(source_idx)
        if gray.shape[:2] != (out_h, out_w):
            gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        channels.append(gray)
    return np.stack(channels, axis=0).astype("float32") / 255.0


def load_stack(path: Path, image_size: tuple[int, int], in_dim: int):
    cv2 = require_cv2()
    np = require_numpy()
    data = np.load(path)
    image = data["image"].astype("float32")
    if image.max() > 1.5:
        image /= 255.0
    if image.shape[1:] != image_size:
        image = np.stack(
            [cv2.resize(channel, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA) for channel in image],
            axis=0,
        )
    if image.shape[0] != in_dim:
        if image.shape[0] < in_dim:
            pad = np.repeat(image[-1:], in_dim - image.shape[0], axis=0)
            image = np.concatenate([image, pad], axis=0)
        else:
            image = image[:in_dim]
    return image


def detections_to_json(detections) -> list[dict]:
    return [
        {
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "box_cxcywh": [det.cx, det.cy, det.width, det.height],
            "box_xyxy": list(det.xyxy),
        }
        for det in detections
    ]


def draw_detections(frame, detections) -> None:
    cv2 = require_cv2()
    h, w = frame.shape[:2]
    colors = {
        0: (0, 0, 255),
        1: (0, 255, 255),
        2: (0, 255, 0),
        3: (255, 0, 255),
        4: (255, 160, 0),
        5: (255, 255, 255),
    }
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        color = colors.get(det.class_id, (255, 255, 255))
        cv2.rectangle(frame, pt1, pt2, color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (pt1[0], max(20, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def infer_tensor(model, tensor, device, score_threshold, iou_threshold, actions_only, max_detections):
    keep_classes = ACTION_CLASS_NAMES.keys() if actions_only else None
    with torch.no_grad():
        output = model(tensor.to(device), decode=True)
        if "action_logits" in output:
            return postprocess_predictions_v2(
                output,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                class_names=EXPORT_CLASS_NAMES,
                keep_classes=keep_classes,
                max_detections=max_detections,
            )[0]
        return postprocess_predictions(
            output,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            class_names=EXPORT_CLASS_NAMES,
            keep_classes=keep_classes,
            max_detections=max_detections,
        )[0]


def infer_stacks(args: argparse.Namespace, model: VballActionDetector, config: dict, device: torch.device) -> dict:
    np = require_numpy()
    image_cfg = config.get("image_size", {})
    image_size = (
        int(args.height or image_cfg.get("height", 432)),
        int(args.width or image_cfg.get("width", 768)),
    )
    results = []
    counter: Counter[str] = Counter()
    for stack_path_str in args.stack:
        stack_path = Path(stack_path_str).expanduser()
        stack = load_stack(stack_path, image_size, model.in_dim)
        tensor = torch.from_numpy(np.expand_dims(stack, axis=0)).float()
        detections = infer_tensor(
            model,
            tensor,
            device,
            args.score_threshold,
            args.iou_threshold,
            args.actions_only,
            args.max_detections,
        )
        for det in detections:
            counter[det.class_name] += 1
        results.append({"stack": str(stack_path), "detections": detections_to_json(detections)})
    return {"inputs": results, "summary": dict(counter)}


def infer_video(args: argparse.Namespace, model: VballActionDetector, config: dict, device: torch.device) -> dict:
    cv2 = require_cv2()
    np = require_numpy()
    video_path = Path(args.video).expanduser()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_cfg = config.get("image_size", {})
    image_size = (
        int(args.height or image_cfg.get("height", 432)),
        int(args.width or image_cfg.get("width", 768)),
    )
    config_offsets = config.get("frame_offsets")
    offsets = parse_frame_offsets(args.frame_offsets) if args.frame_offsets else tuple(
        config_offsets if config_offsets else range(-(model.in_dim // 2), model.in_dim // 2 + 1)
    )
    if len(offsets) != model.in_dim:
        raise ValueError(f"frame offsets count ({len(offsets)}) must equal model in_dim ({model.in_dim})")

    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, fps / max(1, args.stride), (source_w, source_h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video: {args.output_video}")

    cache = FrameCache(capture, limit=max(64, len(offsets) * 4))
    frames = []
    counter: Counter[str] = Counter()
    try:
        for frame_idx in range(0, total_frames, max(1, args.stride)):
            stack = build_video_stack(cache, frame_idx, total_frames, offsets, image_size)
            tensor = torch.from_numpy(np.expand_dims(stack, axis=0)).float()
            detections = infer_tensor(
                model,
                tensor,
                device,
                args.score_threshold,
                args.iou_threshold,
                args.actions_only,
                args.max_detections,
            )
            for det in detections:
                counter[det.class_name] += 1
            frames.append({"frame": frame_idx, "detections": detections_to_json(detections)})
            if writer is not None:
                frame = cache.get_bgr(frame_idx)
                draw_detections(frame, detections)
                writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return {
        "video": str(video_path),
        "total_frames": total_frames,
        "processed_frames": len(frames),
        "stride": args.stride,
        "frames": frames,
        "summary": dict(counter),
    }


def main() -> int:
    args = parse_args()
    if not args.video and not args.stack:
        raise ValueError("Provide --video or at least one --stack")

    device = torch.device(args.device)
    model, config = load_checkpoint(Path(args.checkpoint).expanduser(), device)
    if args.video:
        result = infer_video(args, model, config, device)
    else:
        result = infer_stacks(args, model, config, device)

    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2, ensure_ascii=False)
    print(json.dumps({"output_json": str(output_path), "summary": result.get("summary", {})}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
