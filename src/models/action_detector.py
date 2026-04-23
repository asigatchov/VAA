"""Lightweight anchor-free detector for volleyball actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .constants import ACTION_CLASS_NAMES, EXPORT_CLASS_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - depends on training env
    raise ModuleNotFoundError(
        "PyTorch is required for src.models.action_detector. "
        "Install project dependencies in the training environment first."
    ) from exc


ACTION_CLASS_IDS = tuple(sorted(ACTION_CLASS_NAMES))
HELPER_CLASS_IDS = (4, 5)
HELPER_CLASS_ID_TO_INDEX = {class_id: index for index, class_id in enumerate(HELPER_CLASS_IDS)}
HELPER_INDEX_TO_CLASS_ID = {index: class_id for class_id, index in HELPER_CLASS_ID_TO_INDEX.items()}


@dataclass(frozen=True)
class Detection:
    """One normalized detector output."""

    class_id: int
    class_name: str
    confidence: float
    cx: float
    cy: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            max(0.0, self.cx - half_w),
            max(0.0, self.cy - half_h),
            min(1.0, self.cx + half_w),
            min(1.0, self.cy + half_h),
        )


class DSConvBlock(nn.Module):
    """Depthwise-separable conv block used by the detector backbone."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VballActionDetector(nn.Module):
    """Small anchor-free detector for 4 actions plus optional helpers."""

    def __init__(
        self,
        in_dim: int = 9,
        num_classes: int = 6,
        width_mult: float = 1.0,
        stride: int = 16,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.stride = int(stride)

        def ch(value: int) -> int:
            return max(16, int(round(value * width_mult)))

        c64, c128, c256, c512 = ch(64), ch(128), ch(256), ch(512)
        self.features = nn.Sequential(
            DSConvBlock(self.in_dim, c64),
            DSConvBlock(c64, c64),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c64, c128),
            DSConvBlock(c128, c128),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c128, c256),
            DSConvBlock(c256, c256),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c256, c256),
            DSConvBlock(c256, c256),
            DSConvBlock(c256, c256),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c256, c512),
            DSConvBlock(c512, c512),
            DSConvBlock(c512, c512),
        )
        self.head = nn.Sequential(
            DSConvBlock(c512, c256),
            nn.Conv2d(c256, c256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c256),
            nn.SiLU(inplace=True),
        )
        self.predictor = nn.Conv2d(c256, 4 + 1 + self.num_classes, kernel_size=1)
        nn.init.constant_(self.predictor.bias[4], -4.0)

    def forward(self, x: torch.Tensor, decode: bool = False) -> dict[str, torch.Tensor]:
        feat = self.head(self.features(x))
        raw = self.predictor(feat)
        batch, channels, grid_h, grid_w = raw.shape
        pred = raw.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, channels)
        result = {
            "raw": raw,
            "pred": pred,
            "grid_size": torch.tensor([grid_h, grid_w], device=raw.device),
        }
        if decode:
            result["boxes"] = decode_boxes(pred[..., :4], grid_h, grid_w)
            result["obj_logits"] = pred[..., 4]
            result["class_logits"] = pred[..., 5:]
        return result


class VballActionDetectorV2(nn.Module):
    """Shared detector backbone with separate action/helper classification heads."""

    def __init__(
        self,
        in_dim: int = 9,
        num_classes: int = 6,
        width_mult: float = 1.0,
        stride: int = 16,
    ):
        super().__init__()
        if int(num_classes) != len(EXPORT_CLASS_NAMES):
            raise ValueError(f"V2 expects {len(EXPORT_CLASS_NAMES)} classes, got {num_classes}")
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.stride = int(stride)
        self.num_action_classes = len(ACTION_CLASS_IDS)
        self.num_helper_classes = len(HELPER_CLASS_IDS)

        def ch(value: int) -> int:
            return max(16, int(round(value * width_mult)))

        c64, c128, c256, c512 = ch(64), ch(128), ch(256), ch(512)
        self.features = nn.Sequential(
            DSConvBlock(self.in_dim, c64),
            DSConvBlock(c64, c64),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c64, c128),
            DSConvBlock(c128, c128),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c128, c256),
            DSConvBlock(c256, c256),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c256, c256),
            DSConvBlock(c256, c256),
            DSConvBlock(c256, c256),
            nn.MaxPool2d(2, 2),
            DSConvBlock(c256, c512),
            DSConvBlock(c512, c512),
            DSConvBlock(c512, c512),
        )
        self.shared_head = nn.Sequential(
            DSConvBlock(c512, c256),
            nn.Conv2d(c256, c256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c256),
            nn.SiLU(inplace=True),
        )
        self.box_head = nn.Conv2d(c256, 4, kernel_size=1)
        self.obj_head = nn.Conv2d(c256, 1, kernel_size=1)
        self.action_head = nn.Conv2d(c256, self.num_action_classes, kernel_size=1)
        self.helper_head = nn.Conv2d(c256, self.num_helper_classes, kernel_size=1)
        nn.init.constant_(self.obj_head.bias[0], -4.0)

    def forward(self, x: torch.Tensor, decode: bool = False) -> dict[str, torch.Tensor]:
        feat = self.shared_head(self.features(x))
        box_raw = self.box_head(feat)
        obj_raw = self.obj_head(feat)
        action_raw = self.action_head(feat)
        helper_raw = self.helper_head(feat)

        batch, _, grid_h, grid_w = box_raw.shape
        box_pred = box_raw.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, 4)
        obj_pred = obj_raw.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w)
        action_pred = action_raw.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, self.num_action_classes)
        helper_pred = helper_raw.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, self.num_helper_classes)

        result = {
            "box_raw": box_raw,
            "obj_raw": obj_raw,
            "action_raw": action_raw,
            "helper_raw": helper_raw,
            "boxes_pred": box_pred,
            "obj_pred": obj_pred,
            "action_pred": action_pred,
            "helper_pred": helper_pred,
            "grid_size": torch.tensor([grid_h, grid_w], device=box_raw.device),
        }
        if decode:
            result["boxes"] = decode_boxes(box_pred, grid_h, grid_w)
            result["obj_logits"] = obj_pred
            result["action_logits"] = action_pred
            result["helper_logits"] = helper_pred
        return result


def decode_boxes(regression: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Decode raw regression to normalized ``cx, cy, w, h`` boxes."""

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=regression.device),
        torch.arange(grid_w, device=regression.device),
        indexing="ij",
    )
    grid_x = grid_x.float().flatten()
    grid_y = grid_y.float().flatten()

    cx = (grid_x + torch.sigmoid(regression[..., 0])) / float(grid_w)
    cy = (grid_y + torch.sigmoid(regression[..., 1])) / float(grid_h)
    width = torch.exp(regression[..., 2].clamp(-8.0, 4.0)) / float(grid_w)
    height = torch.exp(regression[..., 3].clamp(-8.0, 4.0)) / float(grid_h)
    return torch.stack([cx, cy, width.clamp(0.0, 1.0), height.clamp(0.0, 1.0)], dim=-1)


def encode_target_box(box: torch.Tensor, grid_h: int, grid_w: int) -> tuple[int, int, torch.Tensor]:
    """Encode one normalized target box for the detector loss."""

    cx, cy, width, height = box.tolist()
    gx = min(grid_w - 1, max(0, int(cx * grid_w)))
    gy = min(grid_h - 1, max(0, int(cy * grid_h)))
    tx = cx * grid_w - gx
    ty = cy * grid_h - gy
    tw = torch.log(torch.tensor(max(width * grid_w, 1e-4), device=box.device))
    th = torch.log(torch.tensor(max(height * grid_h, 1e-4), device=box.device))
    return gy, gx, torch.tensor([tx, ty, tw.item(), th.item()], device=box.device)


def detection_loss(
    raw: torch.Tensor,
    targets: list[torch.Tensor],
    num_classes: int,
    obj_weight: float = 1.0,
    box_weight: float = 5.0,
    cls_weight: float = 1.0,
    obj_pos_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute a compact YOLO-like loss.

    Each target tensor is ``N x 5`` with normalized ``class, cx, cy, w, h``.
    If several targets hit the same grid cell, the smaller-area target is kept.
    """

    batch, _, grid_h, grid_w = raw.shape
    device = raw.device
    pred = raw.permute(0, 2, 3, 1)

    obj_target = torch.zeros((batch, grid_h, grid_w), device=device)
    cls_target = torch.zeros((batch, grid_h, grid_w), dtype=torch.long, device=device)
    box_target = torch.zeros((batch, grid_h, grid_w, 4), device=device)
    area_target = torch.full((batch, grid_h, grid_w), float("inf"), device=device)

    for batch_idx, boxes in enumerate(targets):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device)
        for row in boxes:
            class_id = int(row[0].item())
            if class_id < 0 or class_id >= num_classes:
                continue
            box = row[1:5].clamp(0.0, 1.0)
            if box[2] <= 0.0 or box[3] <= 0.0:
                continue
            gy, gx, encoded = encode_target_box(box, grid_h, grid_w)
            area = box[2] * box[3]
            if area >= area_target[batch_idx, gy, gx]:
                continue
            area_target[batch_idx, gy, gx] = area
            obj_target[batch_idx, gy, gx] = 1.0
            cls_target[batch_idx, gy, gx] = class_id
            box_target[batch_idx, gy, gx] = encoded

    obj_logits = pred[..., 4]
    pos_mask = obj_target > 0.5
    pos_weight_tensor = torch.tensor(float(obj_pos_weight), device=device)
    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits,
        obj_target,
        pos_weight=pos_weight_tensor,
        reduction="mean",
    )

    if pos_mask.any():
        pred_box = pred[..., :4][pos_mask]
        target_box = box_target[pos_mask]
        pred_box_for_loss = torch.stack(
            [
                torch.sigmoid(pred_box[:, 0]),
                torch.sigmoid(pred_box[:, 1]),
                pred_box[:, 2].clamp(-8.0, 4.0),
                pred_box[:, 3].clamp(-8.0, 4.0),
            ],
            dim=-1,
        )
        box_loss = F.smooth_l1_loss(pred_box_for_loss, target_box, reduction="mean")
        if class_weights is not None:
            class_weights = class_weights.to(device=device, dtype=pred.dtype)
        cls_loss = F.cross_entropy(
            pred[..., 5:][pos_mask],
            cls_target[pos_mask],
            weight=class_weights,
            reduction="mean",
        )
    else:
        box_loss = raw.sum() * 0.0
        cls_loss = raw.sum() * 0.0

    total = obj_weight * obj_loss + box_weight * box_loss + cls_weight * cls_loss
    return {"total": total, "obj": obj_loss, "box": box_loss, "cls": cls_loss}


def detection_loss_v2(
    output: dict[str, torch.Tensor],
    targets: list[torch.Tensor],
    obj_weight: float = 1.0,
    box_weight: float = 5.0,
    action_cls_weight: float = 1.0,
    helper_cls_weight: float = 1.0,
    obj_pos_weight: float = 1.0,
    action_class_weights: torch.Tensor | None = None,
    helper_class_weights: torch.Tensor | None = None,
    action_focal_gamma: float = 0.0,
    helper_focal_gamma: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Loss for shared-box/objectness detector with two classification heads."""

    box_raw = output["box_raw"]
    obj_raw = output["obj_raw"]
    action_raw = output["action_raw"]
    helper_raw = output["helper_raw"]

    batch, _, grid_h, grid_w = box_raw.shape
    device = box_raw.device
    box_pred = box_raw.permute(0, 2, 3, 1)
    obj_logits = obj_raw[:, 0, :, :]
    action_logits = action_raw.permute(0, 2, 3, 1)
    helper_logits = helper_raw.permute(0, 2, 3, 1)

    obj_target = torch.zeros((batch, grid_h, grid_w), device=device)
    box_target = torch.zeros((batch, grid_h, grid_w, 4), device=device)
    box_area_target = torch.full((batch, grid_h, grid_w), float("inf"), device=device)
    action_target = torch.zeros((batch, grid_h, grid_w), dtype=torch.long, device=device)
    helper_target = torch.zeros((batch, grid_h, grid_w), dtype=torch.long, device=device)
    action_area_target = torch.full((batch, grid_h, grid_w), float("inf"), device=device)
    helper_area_target = torch.full((batch, grid_h, grid_w), float("inf"), device=device)
    action_mask = torch.zeros((batch, grid_h, grid_w), dtype=torch.bool, device=device)
    helper_mask = torch.zeros((batch, grid_h, grid_w), dtype=torch.bool, device=device)

    for batch_idx, boxes in enumerate(targets):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device)
        for row in boxes:
            class_id = int(row[0].item())
            box = row[1:5].clamp(0.0, 1.0)
            if box[2] <= 0.0 or box[3] <= 0.0:
                continue
            gy, gx, encoded = encode_target_box(box, grid_h, grid_w)
            area = box[2] * box[3]

            if area < box_area_target[batch_idx, gy, gx]:
                box_area_target[batch_idx, gy, gx] = area
                obj_target[batch_idx, gy, gx] = 1.0
                box_target[batch_idx, gy, gx] = encoded

            if class_id in ACTION_CLASS_IDS and area < action_area_target[batch_idx, gy, gx]:
                action_area_target[batch_idx, gy, gx] = area
                action_mask[batch_idx, gy, gx] = True
                action_target[batch_idx, gy, gx] = class_id

            if class_id in HELPER_CLASS_ID_TO_INDEX and area < helper_area_target[batch_idx, gy, gx]:
                helper_area_target[batch_idx, gy, gx] = area
                helper_mask[batch_idx, gy, gx] = True
                helper_target[batch_idx, gy, gx] = HELPER_CLASS_ID_TO_INDEX[class_id]

    pos_weight_tensor = torch.tensor(float(obj_pos_weight), device=device)
    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits,
        obj_target,
        pos_weight=pos_weight_tensor,
        reduction="mean",
    )

    pos_mask = obj_target > 0.5
    if pos_mask.any():
        pred_box = box_pred[pos_mask]
        target_box = box_target[pos_mask]
        pred_box_for_loss = torch.stack(
            [
                torch.sigmoid(pred_box[:, 0]),
                torch.sigmoid(pred_box[:, 1]),
                pred_box[:, 2].clamp(-8.0, 4.0),
                pred_box[:, 3].clamp(-8.0, 4.0),
            ],
            dim=-1,
        )
        box_loss = F.smooth_l1_loss(pred_box_for_loss, target_box, reduction="mean")
    else:
        box_loss = box_raw.sum() * 0.0

    if action_mask.any():
        if action_class_weights is not None:
            action_class_weights = action_class_weights.to(device=device, dtype=action_logits.dtype)
        action_ce = F.cross_entropy(
            action_logits[action_mask],
            action_target[action_mask],
            weight=action_class_weights,
            reduction="none",
        )
        if action_focal_gamma > 0.0:
            action_pt = torch.exp(-action_ce)
            action_cls_loss = (((1.0 - action_pt) ** action_focal_gamma) * action_ce).mean()
        else:
            action_cls_loss = action_ce.mean()
    else:
        action_cls_loss = action_raw.sum() * 0.0

    if helper_mask.any():
        if helper_class_weights is not None:
            helper_class_weights = helper_class_weights.to(device=device, dtype=helper_logits.dtype)
        helper_ce = F.cross_entropy(
            helper_logits[helper_mask],
            helper_target[helper_mask],
            weight=helper_class_weights,
            reduction="none",
        )
        if helper_focal_gamma > 0.0:
            helper_pt = torch.exp(-helper_ce)
            helper_cls_loss = (((1.0 - helper_pt) ** helper_focal_gamma) * helper_ce).mean()
        else:
            helper_cls_loss = helper_ce.mean()
    else:
        helper_cls_loss = helper_raw.sum() * 0.0

    total = (
        obj_weight * obj_loss
        + box_weight * box_loss
        + action_cls_weight * action_cls_loss
        + helper_cls_weight * helper_cls_loss
    )
    return {
        "total": total,
        "obj": obj_loss,
        "box": box_loss,
        "action_cls": action_cls_loss,
        "helper_cls": helper_cls_loss,
    }


def box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-7)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = boxes[..., 2:4] / 2.0
    top_left = boxes[..., 0:2] - half
    bottom_right = boxes[..., 0:2] + half
    return torch.cat([top_left, bottom_right], dim=-1).clamp(0.0, 1.0)


def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = box_iou_xyxy(boxes_xyxy[idx], boxes_xyxy[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep).long()


@torch.no_grad()
def postprocess_predictions(
    model_output: dict[str, torch.Tensor],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    class_names: dict[int, str] | None = None,
    keep_classes: Iterable[int] | None = None,
    max_detections: int | None = 100,
) -> list[list[Detection]]:
    """Convert decoded model output to per-image detections."""

    class_names = class_names or EXPORT_CLASS_NAMES
    keep_set = set(keep_classes) if keep_classes is not None else None
    boxes = model_output["boxes"]
    obj_scores = torch.sigmoid(model_output["obj_logits"])
    class_probs = torch.softmax(model_output["class_logits"], dim=-1)
    if keep_set is not None:
        keep_mask = torch.zeros(class_probs.shape[-1], dtype=torch.bool, device=class_probs.device)
        for class_id in keep_set:
            if 0 <= int(class_id) < class_probs.shape[-1]:
                keep_mask[int(class_id)] = True
        class_probs = class_probs.masked_fill(~keep_mask.view(1, 1, -1), -1.0)
    class_scores, class_ids = class_probs.max(dim=-1)
    scores = obj_scores * class_scores

    batch_detections: list[list[Detection]] = []
    for image_idx in range(boxes.shape[0]):
        mask = scores[image_idx] >= score_threshold
        selected_boxes = boxes[image_idx][mask]
        selected_scores = scores[image_idx][mask]
        selected_classes = class_ids[image_idx][mask]
        boxes_xyxy = cxcywh_to_xyxy(selected_boxes)

        final_indices: list[torch.Tensor] = []
        for class_id in selected_classes.unique():
            class_mask = selected_classes == class_id
            original = torch.nonzero(class_mask, as_tuple=False).flatten()
            kept_local = nms(boxes_xyxy[class_mask], selected_scores[class_mask], iou_threshold)
            if kept_local.numel() > 0:
                final_indices.extend(original[kept_local])

        detections: list[Detection] = []
        sorted_indices = sorted(final_indices, key=lambda i: float(selected_scores[i]), reverse=True)
        if max_detections is not None and max_detections > 0:
            sorted_indices = sorted_indices[:max_detections]
        for idx in sorted_indices:
            class_id = int(selected_classes[idx].item())
            cx, cy, width, height = selected_boxes[idx].tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(selected_scores[idx].item()),
                    cx=float(cx),
                    cy=float(cy),
                    width=float(width),
                    height=float(height),
                )
            )
        batch_detections.append(detections)
    return batch_detections


@torch.no_grad()
def postprocess_predictions_v2(
    model_output: dict[str, torch.Tensor],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    class_names: dict[int, str] | None = None,
    keep_classes: Iterable[int] | None = None,
    max_detections: int | None = 100,
) -> list[list[Detection]]:
    """Postprocess predictions for the dual-head detector."""

    class_names = class_names or EXPORT_CLASS_NAMES
    keep_set = set(keep_classes) if keep_classes is not None else None
    boxes = model_output["boxes"]
    obj_scores = torch.sigmoid(model_output["obj_logits"])
    action_probs = torch.softmax(model_output["action_logits"], dim=-1)
    helper_probs = torch.softmax(model_output["helper_logits"], dim=-1)

    batch_detections: list[list[Detection]] = []
    for image_idx in range(boxes.shape[0]):
        per_box_scores: list[torch.Tensor] = []
        per_box_classes: list[torch.Tensor] = []

        for local_idx, class_id in enumerate(ACTION_CLASS_IDS):
            if keep_set is not None and class_id not in keep_set:
                continue
            per_box_scores.append(obj_scores[image_idx] * action_probs[image_idx, :, local_idx])
            per_box_classes.append(torch.full_like(obj_scores[image_idx], class_id, dtype=torch.long))

        for local_idx, class_id in HELPER_INDEX_TO_CLASS_ID.items():
            if keep_set is not None and class_id not in keep_set:
                continue
            per_box_scores.append(obj_scores[image_idx] * helper_probs[image_idx, :, local_idx])
            per_box_classes.append(torch.full_like(obj_scores[image_idx], class_id, dtype=torch.long))

        if not per_box_scores:
            batch_detections.append([])
            continue

        stacked_scores = torch.stack(per_box_scores, dim=-1)
        stacked_classes = torch.stack(per_box_classes, dim=-1)
        class_scores, best_idx = stacked_scores.max(dim=-1)
        class_ids = stacked_classes.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

        mask = class_scores >= score_threshold
        selected_boxes = boxes[image_idx][mask]
        selected_scores = class_scores[mask]
        selected_classes = class_ids[mask]
        boxes_xyxy = cxcywh_to_xyxy(selected_boxes)

        final_indices: list[torch.Tensor] = []
        for class_id in selected_classes.unique():
            class_mask = selected_classes == class_id
            original = torch.nonzero(class_mask, as_tuple=False).flatten()
            kept_local = nms(boxes_xyxy[class_mask], selected_scores[class_mask], iou_threshold)
            if kept_local.numel() > 0:
                final_indices.extend(original[kept_local])

        detections: list[Detection] = []
        sorted_indices = sorted(final_indices, key=lambda i: float(selected_scores[i]), reverse=True)
        if max_detections is not None and max_detections > 0:
            sorted_indices = sorted_indices[:max_detections]
        for idx in sorted_indices:
            class_id = int(selected_classes[idx].item())
            cx, cy, width, height = selected_boxes[idx].tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(selected_scores[idx].item()),
                    cx=float(cx),
                    cy=float(cy),
                    width=float(width),
                    height=float(height),
                )
            )
        batch_detections.append(detections)
    return batch_detections
