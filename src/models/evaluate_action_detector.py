"""Evaluate VballActionDetector on a prepared dataset split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .action_detector import cxcywh_to_xyxy, postprocess_predictions, postprocess_predictions_v2
from .data import ActionDetectionDataset, collate_detection_batch
from .infer_action_detector import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action detector precision/recall/F1.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--output-json", default="out/action_detector_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-7)


def labels_to_targets(labels: torch.Tensor) -> dict[int, torch.Tensor]:
    targets: dict[int, torch.Tensor] = {}
    if labels.numel() == 0:
        return targets
    boxes = cxcywh_to_xyxy(labels[:, 1:5])
    for class_id in labels[:, 0].long().unique().tolist():
        mask = labels[:, 0].long() == int(class_id)
        targets[int(class_id)] = boxes[mask]
    return targets


def update_counts(
    counts: dict[int, dict[str, int]],
    detections,
    labels: torch.Tensor,
    iou_threshold: float,
) -> None:
    targets_by_class = labels_to_targets(labels.cpu())
    detections_by_class: dict[int, list] = defaultdict(list)
    for detection in detections:
        detections_by_class[detection.class_id].append(detection)

    class_ids = set(targets_by_class) | set(detections_by_class)
    for class_id in class_ids:
        target_boxes = targets_by_class.get(class_id, torch.zeros((0, 4), dtype=torch.float32))
        matched_targets: set[int] = set()
        sorted_detections = sorted(
            detections_by_class.get(class_id, []),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        for detection in sorted_detections:
            det_box = torch.tensor(detection.xyxy, dtype=torch.float32)
            ious = box_iou_one_to_many(det_box, target_boxes)
            if ious.numel() == 0:
                counts[class_id]["fp"] += 1
                continue
            best_iou, best_idx = torch.max(ious, dim=0)
            target_idx = int(best_idx.item())
            if float(best_iou.item()) >= iou_threshold and target_idx not in matched_targets:
                counts[class_id]["tp"] += 1
                matched_targets.add(target_idx)
            else:
                counts[class_id]["fp"] += 1
        counts[class_id]["fn"] += max(0, target_boxes.shape[0] - len(matched_targets))


def summarize_counts(counts: dict[int, dict[str, int]], class_names: dict[int, str]) -> dict:
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for class_id in sorted(class_names):
        row = counts[class_id]
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[str(class_id)] = {
            "name": class_names[class_id],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "per_class": per_class,
        "macro_precision": sum(precisions) / max(1, len(precisions)),
        "macro_recall": sum(recalls) / max(1, len(recalls)),
        "macro_f1": sum(f1s) / max(1, len(f1s)),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_checkpoint(Path(args.checkpoint).expanduser(), device)
    image_cfg = config.get("image_size", {})
    image_size = (
        int(args.height or image_cfg.get("height", 432)),
        int(args.width or image_cfg.get("width", 768)),
    )
    dataset = ActionDetectionDataset(
        args.dataset_dir,
        split=args.split,
        image_size=image_size,
        in_dim=model.in_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )

    raw_class_names = config.get("class_names", {str(i): f"class_{i}" for i in range(model.num_classes)})
    class_names = {int(key): str(value) for key, value in raw_class_names.items() if int(key) < model.num_classes}
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in class_names}

    model.eval()
    with torch.no_grad():
        for images, labels, _metas in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(images, decode=True)
            if "action_logits" in output:
                batch_detections = postprocess_predictions_v2(
                    output,
                    score_threshold=args.score_threshold,
                    iou_threshold=args.iou_threshold,
                    class_names=class_names,
                    max_detections=args.max_detections,
                )
            else:
                batch_detections = postprocess_predictions(
                    output,
                    score_threshold=args.score_threshold,
                    iou_threshold=args.iou_threshold,
                    class_names=class_names,
                    max_detections=args.max_detections,
                )
            for detections, label in zip(batch_detections, labels):
                update_counts(counts, detections, label, args.iou_threshold)

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser()),
        "split": args.split,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "num_samples": len(dataset),
        **summarize_counts(counts, class_names),
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
