"""Dataset preparation and loading for the volleyball action detector."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ACTION_CLASS_NAMES, ACTION_WITH_NOACTION_CLASS_NAMES, EXPORT_CLASS_NAMES, NOACTION_CLASS_ID

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FRAME_OFFSETS = tuple(range(-4, 5))
DEFAULT_ACTION_SUPERFRAME_RADIUS = 1
BALL_DIAMETER_PX = 20
ACTION_CLASS_IDS = {0, 1, 2, 3}
INTERACTION_CROP_SIZE = 224
INTERACTION_CROP_MIN_SIDE_PX = 160
INTERACTION_CROP_CONTEXT_SCALE = 1.6
INTERACTION_BALL_RADIUS_PX = 12
BALL_CIRCUMFERENCE_CM = 67.0
PLAYER_HEIGHT_CM = 190.0
BALL_TOP_PLAYER_CONTEXT_RATIO = 0.10
BALL_BOTTOM_PLAYER_CONTEXT_RATIO = 1.00
BALL_PLAYER_EDGE_PADDING_RATIO = 0.03
DEFAULT_NOACTION_FRAME_OFFSETS = (-18, -12, 12, 18)

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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def normalize_ball_rows(ball_data: Any) -> list[tuple[int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int]] = []
    if not isinstance(ball_data, list):
        return rows
    for row in ball_data:
        try:
            if isinstance(row, dict):
                frame = int(float(row.get("Frame", -1)))
                visibility = int(float(row.get("Visibility", 0)))
                x = int(float(row.get("X", -1)))
                y = int(float(row.get("Y", -1)))
                radius = int(float(row.get("Radius", 0)))
            else:
                frame = int(float(row[0]))
                visibility = int(float(row[1]))
                x = int(float(row[2]))
                y = int(float(row[3]))
                radius = int(float(row[4])) if len(row) >= 5 else 0
        except (IndexError, TypeError, ValueError):
            continue
        rows.append((frame, visibility, x, y, radius))
    return sorted(rows)


def load_ball_positions(project: dict[str, Any]) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    for frame_idx, visibility, x, y, _ in normalize_ball_rows(project.get("ball_data", [])):
        if visibility > 0 and x >= 0 and y >= 0:
            positions[frame_idx] = (x, y)
    return positions


def load_ball_radii(project: dict[str, Any]) -> dict[int, int]:
    radii: dict[int, int] = {}
    for frame_idx, visibility, x, y, radius in normalize_ball_rows(project.get("ball_data", [])):
        if visibility > 0 and x >= 0 and y >= 0 and radius > 0:
            radii[frame_idx] = radius
    return radii


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


def box_cxcywh_to_xyxy(box: tuple[int, float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    _, cx, cy, width, height = box[:5]
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return (
        clamp01(float(cx) - half_w),
        clamp01(float(cy) - half_h),
        clamp01(float(cx) + half_w),
        clamp01(float(cy) + half_h),
    )


def box_xyxy_to_cxcywh(class_id: int, x1: float, y1: float, x2: float, y2: float) -> tuple[int, float, float, float, float] | None:
    x1 = clamp01(x1)
    y1 = clamp01(y1)
    x2 = clamp01(x2)
    y2 = clamp01(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        int(class_id),
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    )


def box_center(box: tuple[int, float, float, float, float] | list[float]) -> tuple[float, float]:
    return float(box[1]), float(box[2])


def squared_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return (float(point_a[0]) - float(point_b[0])) ** 2 + (float(point_a[1]) - float(point_b[1])) ** 2


def choose_closest_box(
    boxes: list[tuple[int, float, float, float, float]],
    reference: tuple[float, float] | tuple[int, float, float, float, float] | list[float] | None,
) -> tuple[int, float, float, float, float] | None:
    if not boxes:
        return None
    if reference is None:
        return boxes[0]
    ref_point = reference if len(reference) == 2 else box_center(reference)  # type: ignore[arg-type]
    return min(boxes, key=lambda box: squared_distance(box_center(box), ref_point))  # type: ignore[arg-type]


def find_nearest_positions(
    positions: dict[int, tuple[int, int]],
    center_frame: int,
    offsets: tuple[int, ...],
    max_gap: int | None = None,
) -> list[tuple[int, int] | None]:
    frames = [center_frame + offset for offset in offsets]
    visible = sorted(positions)
    if not visible:
        return [None for _ in frames]
    max_gap = int(max_gap or max(8, len(offsets)))
    out: list[tuple[int, int] | None] = []
    for frame_idx in frames:
        if frame_idx in positions:
            out.append(positions[frame_idx])
            continue
        nearest = min(visible, key=lambda idx: abs(idx - frame_idx))
        if abs(nearest - frame_idx) > max_gap:
            out.append(None)
            continue
        out.append(positions[nearest])
    return out


def find_nearest_radii(
    radii: dict[int, int],
    center_frame: int,
    offsets: tuple[int, ...],
    max_gap: int | None = None,
) -> list[int | None]:
    frames = [center_frame + offset for offset in offsets]
    visible = sorted(radii)
    if not visible:
        return [None for _ in frames]
    max_gap = int(max_gap or max(8, len(offsets)))
    out: list[int | None] = []
    for frame_idx in frames:
        if frame_idx in radii:
            out.append(radii[frame_idx])
            continue
        nearest = min(visible, key=lambda idx: abs(idx - frame_idx))
        if abs(nearest - frame_idx) > max_gap:
            out.append(None)
            continue
        out.append(radii[nearest])
    return out


def compute_interaction_crop(
    player_box: tuple[int, float, float, float, float] | list[float],
    ball_positions_px: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    min_side_px: int = INTERACTION_CROP_MIN_SIDE_PX,
    context_scale: float = INTERACTION_CROP_CONTEXT_SCALE,
    ball_radius_px: int = INTERACTION_BALL_RADIUS_PX,
) -> tuple[float, float, float, float]:
    px1, py1, px2, py2 = box_cxcywh_to_xyxy(player_box)
    px1 *= image_width
    py1 *= image_height
    px2 *= image_width
    py2 *= image_height
    min_x, min_y, max_x, max_y = px1, py1, px2, py2

    for ball_x, ball_y in ball_positions_px:
        min_x = min(min_x, float(ball_x) - ball_radius_px)
        min_y = min(min_y, float(ball_y) - ball_radius_px)
        max_x = max(max_x, float(ball_x) + ball_radius_px)
        max_y = max(max_y, float(ball_y) + ball_radius_px)

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    side = max(float(min_side_px), width, height) * float(context_scale)
    side = min(max(side, 8.0), float(max(image_width, image_height) * 2))
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def ball_radius_to_cm_per_px(radius_px: float, ball_circumference_cm: float) -> float | None:
    if radius_px <= 0 or ball_circumference_cm <= 0:
        return None
    ball_diameter_cm = ball_circumference_cm / 3.141592653589793
    return ball_diameter_cm / (2.0 * float(radius_px))


def ball_box_radius_px(
    ball_box: tuple[int, float, float, float, float] | list[float] | None,
    image_width: int,
    image_height: int,
) -> float:
    if ball_box is None:
        return 0.0
    return max(0.0, (float(ball_box[3]) * image_width + float(ball_box[4]) * image_height) / 4.0)


def load_annotated_ball_geometry(
    yolo_boxes: dict[str, dict],
    box_classes: dict[int, str],
    frame_width: int,
    frame_height: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    radii: dict[int, int] = {}
    for frame_key, frame_boxes in yolo_boxes.items():
        ball_boxes = remap_project_boxes(
            frame_boxes,
            box_classes,
            frame_width,
            frame_height,
            ball_position=None,
            add_ball_from_tracking=False,
            include_class_ids={5},
        )
        if not ball_boxes:
            continue
        ball_box = max(ball_boxes, key=lambda item: float(item[3]) * float(item[4]))
        frame_idx = int(frame_key)
        positions[frame_idx] = (
            int(round(float(ball_box[1]) * frame_width)),
            int(round(float(ball_box[2]) * frame_height)),
        )
        radii[frame_idx] = max(1, int(round(ball_box_radius_px(ball_box, frame_width, frame_height))))
    return positions, radii


def compute_ball_scaled_interaction_crop(
    player_box: tuple[int, float, float, float, float] | list[float],
    ball_positions_px: list[tuple[float, float]],
    ball_radii_px: list[float] | None,
    image_width: int,
    image_height: int,
    min_side_px: int = INTERACTION_CROP_MIN_SIDE_PX,
    ball_circumference_cm: float = BALL_CIRCUMFERENCE_CM,
    player_height_cm: float = PLAYER_HEIGHT_CM,
    top_player_context_ratio: float = BALL_TOP_PLAYER_CONTEXT_RATIO,
    bottom_player_context_ratio: float = BALL_BOTTOM_PLAYER_CONTEXT_RATIO,
    edge_padding_ratio: float = BALL_PLAYER_EDGE_PADDING_RATIO,
) -> tuple[float, float, float, float]:
    """Build a square action crop whose pixel scale is driven by ball size.

    This mirrors ``inference_openvino_seq_gray_crop.compute_action_crop_rect``:
    the median visible ball radius defines cm/px, and the crop side is sized so
    a full-height player fits around the ball interaction. The annotated player
    box is also enforced so bad ball radius estimates do not cut off the actor.
    """

    radii_for_points = [float(radius) if float(radius) > 0.0 else 0.0 for radius in (ball_radii_px or [])]
    valid_radii = [radius for radius in radii_for_points if radius > 0.0]
    radius_px = float(statistics.median(valid_radii)) if valid_radii else 0.0
    cm_per_px = ball_radius_to_cm_per_px(radius_px, ball_circumference_cm)
    side_by_player = float(player_height_cm / cm_per_px) if cm_per_px else float(min_side_px)
    side_by_player = max(float(min_side_px), side_by_player)

    px1, py1, px2, py2 = box_cxcywh_to_xyxy(player_box)
    px1 *= image_width
    py1 *= image_height
    px2 *= image_width
    py2 *= image_height

    min_x, min_y, max_x, max_y = px1, py1, px2, py2
    for index, (ball_x, ball_y) in enumerate(ball_positions_px):
        radius = (
            radii_for_points[index]
            if index < len(radii_for_points) and radii_for_points[index] > 0.0
            else INTERACTION_BALL_RADIUS_PX
        )
        min_x = min(min_x, float(ball_x) - radius)
        min_y = min(min_y, float(ball_y) - radius)
        max_x = max(max_x, float(ball_x) + radius)
        max_y = max(max_y, float(ball_y) + radius)

    edge_padding = max(2.0, side_by_player * float(edge_padding_ratio))
    min_x -= edge_padding
    min_y -= edge_padding
    max_x += edge_padding
    max_y += edge_padding

    ball_x_values = [float(point[0]) for point in ball_positions_px]
    ball_y_values = [float(point[1]) for point in ball_positions_px]
    if ball_x_values and ball_y_values:
        ball_min_x = min(ball_x_values)
        ball_max_x = max(ball_x_values)
        ball_min_y = min(ball_y_values)
        ball_max_y = max(ball_y_values)
        top_context = side_by_player * float(top_player_context_ratio)
        bottom_context = side_by_player * float(bottom_player_context_ratio)
        side = max(
            side_by_player,
            ball_max_x - ball_min_x + side_by_player,
            ball_max_y - ball_min_y + top_context + bottom_context,
            max_x - min_x,
            max_y - min_y,
        )
        side = max(float(min_side_px), min(float(side), float(max(image_width, image_height) * 2)))
        left = (ball_min_x + ball_max_x) / 2.0 - side / 2.0
        top = ball_min_y - top_context
    else:
        side = max(side_by_player, max_x - min_x, max_y - min_y)
        side = max(float(min_side_px), min(float(side), float(max(image_width, image_height) * 2)))
        left = (min_x + max_x) / 2.0 - side / 2.0
        top = (min_y + max_y) / 2.0 - side / 2.0

    right = left + side
    bottom = top + side
    if min_x < left:
        left -= left - min_x
        right = left + side
    if max_x > right:
        left += max_x - right
        right = left + side
    if min_y < top:
        top -= top - min_y
        bottom = top + side
    if max_y > bottom:
        top += max_y - bottom
        bottom = top + side
    return (left, top, left + side, top + side)


def compute_ball_trajectory_crop(
    ball_positions_px: list[tuple[float, float]],
    ball_radii_px: list[float] | None,
    image_width: int,
    image_height: int,
    min_side_px: int = INTERACTION_CROP_MIN_SIDE_PX,
    ball_circumference_cm: float = BALL_CIRCUMFERENCE_CM,
    player_height_cm: float = PLAYER_HEIGHT_CM,
    top_player_context_ratio: float = BALL_TOP_PLAYER_CONTEXT_RATIO,
    bottom_player_context_ratio: float = BALL_BOTTOM_PLAYER_CONTEXT_RATIO,
) -> tuple[float, float, float, float]:
    """Build the same ball-only action crop used by OpenVINO track inference."""

    if not ball_positions_px:
        side = float(min(image_width, image_height))
        return (0.0, 0.0, side, side)
    valid_radii = [float(radius) for radius in (ball_radii_px or []) if float(radius) > 0.0]
    radius_px = float(statistics.median(valid_radii)) if valid_radii else 0.0
    cm_per_px = ball_radius_to_cm_per_px(radius_px, ball_circumference_cm)
    side_by_player = float(player_height_cm / cm_per_px) if cm_per_px else float(min(image_width, image_height))
    xs = [float(point[0]) for point in ball_positions_px]
    ys = [float(point[1]) for point in ball_positions_px]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    top_context = side_by_player * float(top_player_context_ratio)
    bottom_context = side_by_player * float(bottom_player_context_ratio)
    side = max(
        float(min_side_px),
        side_by_player,
        max_x - min_x + side_by_player,
        max_y - min_y + top_context + bottom_context,
    )
    side = max(1.0, min(side, float(max(image_width, image_height))))
    left = (min_x + max_x) / 2.0 - side / 2.0
    top = min_y - top_context
    left = clamp(left, 0.0, max(0.0, float(image_width) - side))
    top = clamp(top, 0.0, max(0.0, float(image_height) - side))
    return (left, top, left + side, top + side)


def project_box_to_crop(
    label: tuple[int, float, float, float, float] | list[float] | None,
    crop_xyxy_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float] | None:
    if label is None:
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy_px
    crop_w = max(1.0, crop_x2 - crop_x1)
    crop_h = max(1.0, crop_y2 - crop_y1)
    box_x1, box_y1, box_x2, box_y2 = box_cxcywh_to_xyxy(label)
    box_x1 *= image_width
    box_y1 *= image_height
    box_x2 *= image_width
    box_y2 *= image_height
    proj = box_xyxy_to_cxcywh(
        int(label[0]),
        (box_x1 - crop_x1) / crop_w,
        (box_y1 - crop_y1) / crop_h,
        (box_x2 - crop_x1) / crop_w,
        (box_y2 - crop_y1) / crop_h,
    )
    return proj


def project_point_to_crop(
    point_px: tuple[float, float] | None,
    crop_xyxy_px: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    if point_px is None:
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy_px
    crop_w = max(1.0, crop_x2 - crop_x1)
    crop_h = max(1.0, crop_y2 - crop_y1)
    return (
        clamp((float(point_px[0]) - crop_x1) / crop_w, 0.0, 1.0),
        clamp((float(point_px[1]) - crop_y1) / crop_h, 0.0, 1.0),
    )


def crop_and_resize_channel(channel, crop_xyxy_px: tuple[float, float, float, float], crop_size: int):
    cv2 = require_cv2()
    np = require_numpy()
    height, width = channel.shape[:2]
    x1, y1, x2, y2 = crop_xyxy_px
    x1i = int(round(x1))
    y1i = int(round(y1))
    x2i = int(round(x2))
    y2i = int(round(y2))
    pad_l = max(0, -x1i)
    pad_t = max(0, -y1i)
    pad_r = max(0, x2i - width)
    pad_b = max(0, y2i - height)
    if pad_l or pad_t or pad_r or pad_b:
        channel = np.pad(channel, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        x1i += pad_l
        x2i += pad_l
        y1i += pad_t
        y2i += pad_t
    crop = channel[y1i:y2i, x1i:x2i]
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return crop


def select_interaction_sample(
    yolo_boxes: dict[str, dict],
    box_classes: dict[int, str],
    action_class_id: int,
    source_frame: int,
    frame_width: int,
    frame_height: int,
    ball_positions: dict[int, tuple[int, int]],
) -> tuple[
    tuple[int, float, float, float, float] | None,
    tuple[int, float, float, float, float] | None,
    tuple[int, float, float, float, float] | None,
]:
    source_boxes = yolo_boxes.get(str(source_frame), {})
    action_boxes = remap_project_boxes(
        source_boxes,
        box_classes,
        frame_width,
        frame_height,
        None,
        add_ball_from_tracking=False,
        include_class_ids={action_class_id},
    )
    if not action_boxes:
        return None, None, None
    ball_boxes = remap_project_boxes(
        source_boxes,
        box_classes,
        frame_width,
        frame_height,
        ball_positions.get(source_frame),
        add_ball_from_tracking=True,
        include_class_ids={5},
    )
    player_boxes = remap_project_boxes(
        source_boxes,
        box_classes,
        frame_width,
        frame_height,
        None,
        add_ball_from_tracking=False,
        include_class_ids={4},
    )
    action_label = choose_closest_box(action_boxes, choose_closest_box(ball_boxes, action_boxes[0]) or action_boxes[0])
    ball_label = choose_closest_box(ball_boxes, action_label)
    player_label = choose_closest_box(player_boxes, ball_label or action_label) or action_label
    return action_label, player_label, ball_label


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

        project_name = str(project.get("name") or project_path.stem)
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


def prepare_interaction_crops(
    project_paths: list[Path],
    output_dir: Path,
    image_size: tuple[int, int] = (432, 768),
    frame_offsets: tuple[int, ...] = DEFAULT_FRAME_OFFSETS,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    crop_size: int = INTERACTION_CROP_SIZE,
    min_side_px: int = INTERACTION_CROP_MIN_SIDE_PX,
    context_scale: float = INTERACTION_CROP_CONTEXT_SCALE,
    crop_scale_mode: str = "ball",
    ball_circumference_cm: float = BALL_CIRCUMFERENCE_CM,
    player_height_cm: float = PLAYER_HEIGHT_CM,
    ball_source: str = "annotations",
    include_noaction: bool = False,
    noaction_frame_offsets: tuple[int, ...] = DEFAULT_NOACTION_FRAME_OFFSETS,
    noaction_ratio: float = 1.0,
) -> dict[str, Any]:
    np = require_numpy()
    all_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for project_path in project_paths:
        project = read_json(project_path)
        video_path = Path(project["video_path"]).expanduser()
        annotations = project.get("annotations", {})
        yolo_boxes = annotations.get("yolo_boxes", {})
        box_classes = {int(key): value for key, value in annotations.get("box_classes", {}).items()}
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

        annotated_ball_positions, annotated_ball_radii = load_annotated_ball_geometry(
            yolo_boxes,
            box_classes,
            frame_width,
            frame_height,
        )
        tracking_ball_positions = load_ball_positions(project)
        tracking_ball_radii = load_ball_radii(project)
        if ball_source == "tracking":
            ball_positions = tracking_ball_positions
            ball_radii = tracking_ball_radii
        elif ball_source == "annotations-then-tracking":
            ball_positions = {**tracking_ball_positions, **annotated_ball_positions}
            ball_radii = {**tracking_ball_radii, **annotated_ball_radii}
        else:
            ball_positions = annotated_ball_positions
            ball_radii = annotated_ball_radii

        project_name = str(project.get("name") or project_path.stem)
        action_mask_sources = build_action_mask_sources(yolo_boxes, box_classes, total_frames)
        export_frames = build_export_frames(yolo_boxes, total_frames, box_classes, frame_offsets)
        noaction_frames: list[int] = []
        if include_noaction:
            action_source_frames = sorted(
                int(frame_key)
                for frame_key, frame_boxes in yolo_boxes.items()
                if frame_boxes and frame_action_class_ids(frame_boxes, box_classes)
            )
            noaction_candidates: list[int] = []
            for source_frame in action_source_frames:
                for offset in noaction_frame_offsets:
                    frame_idx = source_frame + int(offset)
                    if frame_idx < 0 or frame_idx >= total_frames:
                        continue
                    selected = select_temporal_action_sources(
                        frame_idx,
                        frame_offsets,
                        total_frames,
                        action_mask_sources,
                        action_overlap_threshold(frame_offsets),
                    )
                    if selected:
                        continue
                    temporal_ball_points = find_nearest_positions(ball_positions, frame_idx, frame_offsets)
                    visible_ball_points = [point for point in temporal_ball_points if point is not None]
                    if len(visible_ball_points) < max(3, len(frame_offsets) // 2):
                        continue
                    noaction_candidates.append(frame_idx)
            noaction_frames = sorted(set(noaction_candidates))
            if noaction_ratio >= 0.0:
                max_noaction = int(round(len(export_frames) * float(noaction_ratio)))
                if max_noaction and len(noaction_frames) > max_noaction:
                    project_seed = sum(ord(char) for char in project_name)
                    rng = random.Random(seed + project_seed)
                    rng.shuffle(noaction_frames)
                    noaction_frames = sorted(noaction_frames[:max_noaction])
        required = collect_required_indices(sorted(set(export_frames) | set(noaction_frames)), frame_offsets, total_frames)
        gray_frames, video_info = extract_gray_frames(video_path, required)
        resized_h, resized_w = image_size

        samples_for_project = 0
        for frame_idx in export_frames:
            selected_sources = select_temporal_action_sources(
                frame_idx,
                frame_offsets,
                total_frames,
                action_mask_sources,
                action_overlap_threshold(frame_offsets),
            )
            if len(selected_sources) != 1:
                continue
            action_class_id, source_frame = next(iter(sorted(selected_sources.items())))
            action_label, player_label, ball_label = select_interaction_sample(
                yolo_boxes,
                box_classes,
                action_class_id,
                source_frame,
                frame_width,
                frame_height,
                ball_positions,
            )
            if action_label is None or player_label is None:
                continue
            if ball_source == "annotations" and ball_label is None:
                continue

            stack = build_stack(gray_frames, frame_idx, frame_offsets, total_frames, image_size)
            temporal_ball_points = find_nearest_positions(ball_positions, frame_idx, frame_offsets)
            temporal_ball_radii = find_nearest_radii(ball_radii, frame_idx, frame_offsets)
            radius_scale = (resized_w / max(1, frame_width) + resized_h / max(1, frame_height)) / 2.0
            resized_ball_points = [
                (
                    (float(point[0]) / max(1, frame_width)) * resized_w,
                    (float(point[1]) / max(1, frame_height)) * resized_h,
                )
                for point in temporal_ball_points
                if point is not None
            ]
            resized_ball_radii = [
                float(radius or 0) * radius_scale
                for point, radius in zip(temporal_ball_points, temporal_ball_radii, strict=True)
                if point is not None
            ]
            if ball_label is not None:
                resized_ball_points.append(
                    (
                        float(ball_label[1]) * resized_w,
                        float(ball_label[2]) * resized_h,
                    )
                )
                resized_ball_radii.append(ball_box_radius_px(ball_label, resized_w, resized_h))
            if not resized_ball_points:
                continue

            if crop_scale_mode == "ball":
                crop_xyxy_px = compute_ball_scaled_interaction_crop(
                    player_label,
                    resized_ball_points,
                    resized_ball_radii,
                    resized_w,
                    resized_h,
                    min_side_px=min_side_px,
                    ball_circumference_cm=ball_circumference_cm,
                    player_height_cm=player_height_cm,
                )
            else:
                crop_xyxy_px = compute_interaction_crop(
                    player_label,
                    resized_ball_points,
                    resized_w,
                    resized_h,
                    min_side_px=min_side_px,
                    context_scale=context_scale,
                )
            crop_stack = np.stack(
                [crop_and_resize_channel(channel, crop_xyxy_px, crop_size) for channel in stack],
                axis=0,
            ).astype("uint8")

            projected_labels = []
            for label in (action_label, player_label, ball_label):
                projected = project_box_to_crop(label, crop_xyxy_px, resized_w, resized_h)
                if projected is not None:
                    projected_labels.append(projected)

            projected_temporal_ball_positions = [
                project_point_to_crop(point, crop_xyxy_px)
                if point is not None
                else None
                for point in [
                    (
                        (float(ball_point[0]) / max(1, frame_width)) * resized_w,
                        (float(ball_point[1]) / max(1, frame_height)) * resized_h,
                    )
                    if ball_point is not None
                    else None
                    for ball_point in temporal_ball_points
                ]
            ]
            temporal_xy = [
                projected if projected is not None else (0.0, 0.0)
                for projected in projected_temporal_ball_positions
            ]
            temporal_valid = [1 if projected is not None else 0 for projected in projected_temporal_ball_positions]
            temporal_radii = [
                float(radius or 0) * radius_scale if point is not None else 0.0
                for point, radius in zip(temporal_ball_points, temporal_ball_radii, strict=True)
            ]
            crop_side_px = float(crop_xyxy_px[2] - crop_xyxy_px[0])

            stem = f"{project_name}_{frame_idx:06d}_c{action_class_id}"
            image_rel = Path("all") / "images" / f"{stem}.npz"
            label_rel = Path("all") / "labels" / f"{stem}.txt"
            image_path = output_dir / image_rel
            label_path = output_dir / label_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                image_path,
                image=crop_stack,
                action_class_id=np.asarray(action_class_id, dtype="int64"),
                crop_box_xyxy=np.asarray(crop_xyxy_px, dtype="float32"),
                crop_side_px=np.asarray(crop_side_px, dtype="float32"),
                temporal_ball_positions=np.asarray(temporal_xy, dtype="float32"),
                temporal_ball_valid=np.asarray(temporal_valid, dtype="uint8"),
                temporal_ball_radii_px=np.asarray(temporal_radii, dtype="float32"),
                source_frame=np.asarray(source_frame, dtype="int64"),
                center_frame=np.asarray(frame_idx, dtype="int64"),
            )
            write_label_file(label_path, projected_labels)
            all_records.append(
                {
                    "image_path": str(image_rel),
                    "label_path": str(label_rel),
                    "source_project": str(project_path),
                    "source_video": str(video_path),
                    "source_frame": frame_idx,
                    "action_source_frame": source_frame,
                    "action_class_id": action_class_id,
                    "num_labels": len(projected_labels),
                    "temporal_ball_visible": int(sum(temporal_valid)),
                    "crop_side_px": crop_side_px,
                    "crop_scale_mode": crop_scale_mode,
                    "ball_source": ball_source,
                }
            )
            samples_for_project += 1

        noaction_samples_for_project = 0
        for frame_idx in noaction_frames:
            stack = build_stack(gray_frames, frame_idx, frame_offsets, total_frames, image_size)
            temporal_ball_points = find_nearest_positions(ball_positions, frame_idx, frame_offsets)
            temporal_ball_radii = find_nearest_radii(ball_radii, frame_idx, frame_offsets)
            radius_scale = (resized_w / max(1, frame_width) + resized_h / max(1, frame_height)) / 2.0
            resized_ball_points = [
                (
                    (float(point[0]) / max(1, frame_width)) * resized_w,
                    (float(point[1]) / max(1, frame_height)) * resized_h,
                )
                for point in temporal_ball_points
                if point is not None
            ]
            resized_ball_radii = [
                float(radius or 0) * radius_scale
                for point, radius in zip(temporal_ball_points, temporal_ball_radii, strict=True)
                if point is not None
            ]
            if not resized_ball_points:
                continue

            crop_xyxy_px = compute_ball_trajectory_crop(
                resized_ball_points,
                resized_ball_radii,
                resized_w,
                resized_h,
                min_side_px=min_side_px,
                ball_circumference_cm=ball_circumference_cm,
                player_height_cm=player_height_cm,
            )
            crop_stack = np.stack(
                [crop_and_resize_channel(channel, crop_xyxy_px, crop_size) for channel in stack],
                axis=0,
            ).astype("uint8")

            projected_temporal_ball_positions = [
                project_point_to_crop(point, crop_xyxy_px)
                if point is not None
                else None
                for point in [
                    (
                        (float(ball_point[0]) / max(1, frame_width)) * resized_w,
                        (float(ball_point[1]) / max(1, frame_height)) * resized_h,
                    )
                    if ball_point is not None
                    else None
                    for ball_point in temporal_ball_points
                ]
            ]
            temporal_xy = [
                projected if projected is not None else (0.0, 0.0)
                for projected in projected_temporal_ball_positions
            ]
            temporal_valid = [1 if projected is not None else 0 for projected in projected_temporal_ball_positions]
            temporal_radii = [
                float(radius or 0) * radius_scale if point is not None else 0.0
                for point, radius in zip(temporal_ball_points, temporal_ball_radii, strict=True)
            ]
            crop_side_px = float(crop_xyxy_px[2] - crop_xyxy_px[0])

            projected_labels = []
            center_projection = projected_temporal_ball_positions[len(projected_temporal_ball_positions) // 2]
            center_radius = temporal_radii[len(temporal_radii) // 2] if temporal_radii else 0.0
            if center_projection is not None and center_radius > 0.0:
                projected_radius = clamp(center_radius / max(1.0, crop_side_px), 0.0, 0.5)
                projected_labels.append(
                    (
                        5,
                        float(center_projection[0]),
                        float(center_projection[1]),
                        projected_radius * 2.0,
                        projected_radius * 2.0,
                    )
                )

            stem = f"{project_name}_{frame_idx:06d}_c{NOACTION_CLASS_ID}_noaction"
            image_rel = Path("all") / "images" / f"{stem}.npz"
            label_rel = Path("all") / "labels" / f"{stem}.txt"
            image_path = output_dir / image_rel
            label_path = output_dir / label_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                image_path,
                image=crop_stack,
                action_class_id=np.asarray(NOACTION_CLASS_ID, dtype="int64"),
                crop_box_xyxy=np.asarray(crop_xyxy_px, dtype="float32"),
                crop_side_px=np.asarray(crop_side_px, dtype="float32"),
                temporal_ball_positions=np.asarray(temporal_xy, dtype="float32"),
                temporal_ball_valid=np.asarray(temporal_valid, dtype="uint8"),
                temporal_ball_radii_px=np.asarray(temporal_radii, dtype="float32"),
                source_frame=np.asarray(frame_idx, dtype="int64"),
                center_frame=np.asarray(frame_idx, dtype="int64"),
            )
            write_label_file(label_path, projected_labels)
            all_records.append(
                {
                    "image_path": str(image_rel),
                    "label_path": str(label_rel),
                    "source_project": str(project_path),
                    "source_video": str(video_path),
                    "source_frame": frame_idx,
                    "action_source_frame": None,
                    "action_class_id": NOACTION_CLASS_ID,
                    "num_labels": len(projected_labels),
                    "temporal_ball_visible": int(sum(temporal_valid)),
                    "crop_side_px": crop_side_px,
                    "crop_scale_mode": "ball-trajectory",
                    "ball_source": ball_source,
                    "noaction_source": "before_after_action_trajectory",
                }
            )
            noaction_samples_for_project += 1

        write_json(
            output_dir / "projects" / f"{project_name}.json",
            {
                "project_path": str(project_path),
                "video_path": str(video_path),
                "video_info": video_info,
                "num_samples": samples_for_project,
                "num_noaction_samples": noaction_samples_for_project,
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

    class_names = ACTION_WITH_NOACTION_CLASS_NAMES if include_noaction else ACTION_CLASS_NAMES
    class_counts = {str(class_id): 0 for class_id in class_names}
    for record in all_records:
        class_counts[str(record["action_class_id"])] += 1

    summary = {
        "format": "vaa-action-interaction-crop-v1",
        "num_samples": len(all_records),
        "image_size": {"height": crop_size, "width": crop_size},
        "source_image_size": {"height": image_size[0], "width": image_size[1]},
        "frame_offsets": list(frame_offsets),
        "in_dim": len(frame_offsets),
        "num_classes": len(class_names),
        "class_names": class_names,
        "crop_min_side_px": min_side_px,
        "crop_context_scale": context_scale,
        "splits": {name: len(records) for name, records in split_map.items()},
        "class_counts": class_counts,
        "crop_scale_mode": crop_scale_mode,
        "ball_source": ball_source,
        "include_noaction": include_noaction,
        "noaction_frame_offsets": list(noaction_frame_offsets),
        "noaction_ratio": noaction_ratio,
        "ball_circumference_cm": ball_circumference_cm,
        "player_height_cm": player_height_cm,
        "crop_edge_padding_ratio": BALL_PLAYER_EDGE_PADDING_RATIO,
        "label_schema": "txt labels contain projected action/player/ball boxes inside crop; npz stores temporal ball positions",
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
    parser.add_argument("--dataset-kind", choices=("stack", "interaction-crop"), default="stack")
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frame-offsets", default=",".join(str(x) for x in DEFAULT_FRAME_OFFSETS))
    parser.add_argument("--splits", default="0.8,0.1,0.1", help="train,valid,test ratios.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ball-from-tracking", action="store_true")
    parser.add_argument("--action-only", action="store_true", help="Keep only action classes 0..3 in labels.")
    parser.add_argument("--crop-size", type=int, default=INTERACTION_CROP_SIZE)
    parser.add_argument("--crop-min-side", type=int, default=INTERACTION_CROP_MIN_SIDE_PX)
    parser.add_argument("--crop-context-scale", type=float, default=INTERACTION_CROP_CONTEXT_SCALE)
    parser.add_argument(
        "--crop-scale-mode",
        choices=("ball", "union"),
        default="ball",
        help="interaction-crop sizing: ball uses ball radius/player height; union keeps legacy player+ball bbox scale.",
    )
    parser.add_argument("--ball-circumference-cm", type=float, default=BALL_CIRCUMFERENCE_CM)
    parser.add_argument("--player-height-cm", type=float, default=PLAYER_HEIGHT_CM)
    parser.add_argument(
        "--interaction-ball-source",
        choices=("annotations", "tracking", "annotations-then-tracking"),
        default="annotations",
        help="Ball source for interaction-crop. Use annotations to avoid false CSV ball detections.",
    )
    parser.add_argument(
        "--include-noaction",
        action="store_true",
        help="Add noaction interaction-crop samples before/after annotated actions using ball trajectory.",
    )
    parser.add_argument(
        "--noaction-frame-offsets",
        default=",".join(str(value) for value in DEFAULT_NOACTION_FRAME_OFFSETS),
        help="Frame offsets from annotated action center used as noaction candidates.",
    )
    parser.add_argument(
        "--noaction-ratio",
        type=float,
        default=1.0,
        help="Maximum noaction/action export-frame ratio per project. Use a negative value for no cap.",
    )
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

    if args.dataset_kind == "interaction-crop":
        summary = prepare_interaction_crops(
            project_paths=project_paths,
            output_dir=Path(args.output_dir).expanduser().resolve(),
            image_size=(args.height, args.width),
            frame_offsets=parse_frame_offsets(args.frame_offsets),
            splits=parse_splits(args.splits),
            seed=args.seed,
            crop_size=args.crop_size,
            min_side_px=args.crop_min_side,
            context_scale=args.crop_context_scale,
            crop_scale_mode=args.crop_scale_mode,
            ball_circumference_cm=args.ball_circumference_cm,
            player_height_cm=args.player_height_cm,
            ball_source=args.interaction_ball_source,
            include_noaction=args.include_noaction,
            noaction_frame_offsets=parse_frame_offsets(args.noaction_frame_offsets),
            noaction_ratio=args.noaction_ratio,
        )
    else:
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
