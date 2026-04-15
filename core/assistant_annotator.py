"""Assistant clip annotation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import rfdetr
from rfdetr.assets.coco_classes import COCO_CLASSES

from core.annotation_manager import AnnotationManager


PERSON_CLASS_NAME = "person"
BALL_CLASS_NAME = "sports ball"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class AssistantAnnotator:
    """Generate a short action clip around a clicked player with RF-DETR Medium."""

    clip_radius: int = 4
    crop_size_px: Tuple[int, int] = (960, 540)
    model_variant: str = "medium"
    resolution: int = 640
    person_threshold: float = 0.55
    ball_threshold: float = 0.12
    _model: object | None = field(default=None, init=False, repr=False)

    def annotate_clip(
        self,
        processor,
        annotations: AnnotationManager,
        ball_lookup: Dict[int, list],
        center_frame: int,
        click_center: Tuple[float, float],
        action_class_id: int,
        player_class_id: int,
        ball_class_id: int,
        replace_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Dict[str, object]:
        """Annotate a 9-frame clip centered on center_frame."""
        del ball_lookup

        total_frames = int(getattr(processor, "total_frames", 0))
        if total_frames <= 0:
            return {"ok": False, "reason": "no_frames"}

        center_image = processor.get_frame(int(center_frame))
        if center_image is None:
            return {"ok": False, "reason": "no_center_frame"}

        frame_height, frame_width = center_image.shape[:2]
        start_frame = max(0, int(center_frame) - self.clip_radius)
        end_frame = min(total_frames - 1, int(center_frame) + self.clip_radius)

        player_detection = self._pick_player_detection(center_image, click_center)
        if player_detection is None:
            return {"ok": False, "reason": "no_player_detection"}

        initial_player_center = self._center_from_xyxy(player_detection.xyxy, frame_width, frame_height)

        if replace_existing:
            annotations.clear_boxes_by_class_ids_in_range(
                start_frame,
                end_frame,
                {int(action_class_id), int(player_class_id), int(ball_class_id)},
            )

        created_boxes = 0
        ball_points: Dict[int, Tuple[int, int]] = {}
        previous_player_center = initial_player_center
        action_start = max(start_frame, int(center_frame) - 1)
        action_end = min(end_frame, int(center_frame) + 1)
        frame_indices = list(range(start_frame, end_frame + 1))
        for progress_index, frame_idx in enumerate(frame_indices, start=1):
            if progress_callback is not None:
                progress_callback(progress_index - 1, len(frame_indices), frame_idx)
            frame_image = processor.get_frame(frame_idx)
            if frame_image is None:
                continue
            frame_height, frame_width = frame_image.shape[:2]

            frame_detections = self._predict_in_crop(
                frame_image,
                previous_player_center,
                threshold=min(self.person_threshold, self.ball_threshold),
            )
            player_detection_frame = self._pick_player_detection_from_detections(
                frame_detections,
                previous_player_center,
                frame_width,
                frame_height,
            )
            if player_detection_frame is None:
                continue

            player_center_frame = self._center_from_xyxy(player_detection_frame.xyxy, frame_width, frame_height)
            player_size = self._size_from_xyxy(player_detection_frame.xyxy, frame_width, frame_height)
            previous_player_center = player_center_frame

            player_box = self._make_box(int(player_class_id), player_center_frame, player_size)
            annotations.add_yolo_box(frame_idx, player_box)
            created_boxes += 1

            ball_box = None
            ball_detection_frame = self._pick_ball_detection_from_detections(frame_detections)
            if ball_detection_frame is not None:
                ball_center_frame = self._center_from_xyxy(ball_detection_frame.xyxy, frame_width, frame_height)
                ball_size = self._size_from_xyxy(ball_detection_frame.xyxy, frame_width, frame_height)
                ball_box = self._make_box(int(ball_class_id), ball_center_frame, ball_size)
                annotations.add_yolo_box(frame_idx, ball_box)
                created_boxes += 1
                ball_points[frame_idx] = self._normalized_to_source_point(
                    ball_center_frame,
                    processor,
                    frame_width,
                    frame_height,
                )

            if action_start <= frame_idx <= action_end:
                action_box = self._merge_boxes(int(action_class_id), player_box, ball_box)
                annotations.add_yolo_box(frame_idx, action_box)
                created_boxes += 1

        if progress_callback is not None:
            progress_callback(len(frame_indices), len(frame_indices), end_frame)

        created_rally = self._ensure_clip_rally(annotations, processor, start_frame, end_frame)
        return {
            "ok": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": end_frame - start_frame + 1,
            "boxes_created": created_boxes,
            "ball_points": ball_points,
            "created_rally": created_rally,
            "model_variant": self.model_variant,
        }

    def _get_model(self):
        if self._model is not None:
            return self._model

        class_name = {
            "nano": "RFDETRNano",
            "medium": "RFDETRMedium",
            "base": "RFDETRBase",
        }.get(self.model_variant, "RFDETRMedium")
        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            raise RuntimeError(f"RF-DETR variant is unavailable: {self.model_variant}")

        self._model = model_cls(resolution=self.resolution)
        self._model.optimize_for_inference()
        return self._model

    def _predict(self, frame_bgr: np.ndarray, threshold: float) -> list[Detection]:
        model = self._get_model()
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

    def _predict_in_crop(
        self,
        frame_bgr: np.ndarray,
        center_norm: Tuple[float, float],
        threshold: float,
    ) -> list[Detection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._build_crop_rect(
            frame_width=frame_width,
            frame_height=frame_height,
            center_norm=center_norm,
            crop_size_px=self.crop_size_px,
        )
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        detections = self._predict(crop, threshold=threshold)
        remapped: list[Detection] = []
        for detection in detections:
            dx1, dy1, dx2, dy2 = detection.xyxy
            remapped.append(
                Detection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    xyxy=(dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1),
                )
            )
        return remapped

    def _pick_player_detection(
        self,
        frame_bgr: np.ndarray,
        click_center: Tuple[float, float],
    ) -> Optional[Detection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        detections = self._predict_in_crop(
            frame_bgr,
            click_center,
            threshold=min(self.person_threshold, self.ball_threshold),
        )
        click_x = float(click_center[0]) * frame_width
        click_y = float(click_center[1]) * frame_height

        return self._pick_player_detection_from_detections(
            detections,
            (
                float(click_x) / max(1, frame_width),
                float(click_y) / max(1, frame_height),
            ),
            frame_width,
            frame_height,
        )

    def _pick_player_detection_from_detections(
        self,
        detections: list[Detection],
        reference_center_norm: Tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Detection]:
        candidates = [
            detection
            for detection in detections
            if detection.class_name == PERSON_CLASS_NAME and detection.confidence >= self.person_threshold
        ]
        if not candidates:
            return None

        reference_x = float(reference_center_norm[0]) * frame_width
        reference_y = float(reference_center_norm[1]) * frame_height

        def sort_key(item: Detection) -> tuple[float, float]:
            center_x, center_y = self._xyxy_center_px(item.xyxy)
            distance = (center_x - reference_x) ** 2 + (center_y - reference_y) ** 2
            return (distance, -item.confidence)

        return min(candidates, key=sort_key)

    def _pick_ball_detection_from_detections(self, detections: list[Detection]) -> Optional[Detection]:
        candidates = [
            detection
            for detection in detections
            if detection.class_name == BALL_CLASS_NAME and detection.confidence >= self.ball_threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _ensure_clip_rally(self, annotations: AnnotationManager, processor, start_frame: int, end_frame: int) -> bool:
        for rally in annotations.rallies:
            if int(rally.get("start_frame", -1)) == start_frame and int(rally.get("end_frame", -1)) == end_frame:
                return False
            if int(rally.get("start_frame", -1)) <= start_frame and int(rally.get("end_frame", -1)) >= end_frame:
                return False

        started = annotations.start_rally(start_frame, getattr(processor, "fps", 30.0))
        if not started:
            return False
        return annotations.end_rally(end_frame) is not None

    @staticmethod
    def _size_from_xyxy(
        xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        width = max(0.0, min(1.0, float(x2 - x1) / max(1, frame_width)))
        height = max(0.0, min(1.0, float(y2 - y1) / max(1, frame_height)))
        return (width, height)

    @staticmethod
    def _center_from_xyxy(
        xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        center_x, center_y = AssistantAnnotator._xyxy_center_px(xyxy)
        return AssistantAnnotator._clamp_point(
            (float(center_x) / max(1, frame_width), float(center_y) / max(1, frame_height))
        )

    @staticmethod
    def _xyxy_center_px(xyxy: tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _normalized_to_source_point(
        center: Tuple[float, float],
        processor,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int]:
        source_width = max(1, int(getattr(processor, "width", frame_width)))
        source_height = max(1, int(getattr(processor, "height", frame_height)))
        return (
            int(round(float(center[0]) * source_width)),
            int(round(float(center[1]) * source_height)),
        )

    @staticmethod
    def _make_box(class_id: int, center: Tuple[float, float], size: Tuple[float, float]) -> Tuple[int, float, float, float, float]:
        width = max(0.0, min(1.0, float(size[0])))
        height = max(0.0, min(1.0, float(size[1])))
        x_center = max(width / 2.0, min(1.0 - width / 2.0, float(center[0])))
        y_center = max(height / 2.0, min(1.0 - height / 2.0, float(center[1])))
        return (int(class_id), x_center, y_center, width, height)

    @staticmethod
    def _merge_boxes(
        class_id: int,
        primary_box: Tuple[int, float, float, float, float],
        secondary_box: Optional[Tuple[int, float, float, float, float]],
    ) -> Tuple[int, float, float, float, float]:
        if secondary_box is None:
            return (int(class_id), primary_box[1], primary_box[2], primary_box[3], primary_box[4])

        _, px, py, pw, ph = primary_box
        _, sx, sy, sw, sh = secondary_box

        p_x1 = px - pw / 2.0
        p_y1 = py - ph / 2.0
        p_x2 = px + pw / 2.0
        p_y2 = py + ph / 2.0

        s_x1 = sx - sw / 2.0
        s_y1 = sy - sh / 2.0
        s_x2 = sx + sw / 2.0
        s_y2 = sy + sh / 2.0

        x1 = max(0.0, min(1.0, min(p_x1, s_x1)))
        y1 = max(0.0, min(1.0, min(p_y1, s_y1)))
        x2 = max(0.0, min(1.0, max(p_x2, s_x2)))
        y2 = max(0.0, min(1.0, max(p_y2, s_y2)))

        return (
            int(class_id),
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
        )

    @staticmethod
    def _build_crop_rect(
        frame_width: int,
        frame_height: int,
        center_norm: Tuple[float, float],
        crop_size_px: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        crop_width = min(max(1, int(crop_size_px[0])), frame_width)
        crop_height = min(max(1, int(crop_size_px[1])), frame_height)
        center_x = int(round(center_norm[0] * frame_width))
        center_y = int(round(center_norm[1] * frame_height))

        x1 = center_x - crop_width // 2
        y1 = center_y - crop_height // 2
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > frame_width:
            x1 -= (x2 - frame_width)
            x2 = frame_width
        if y2 > frame_height:
            y1 -= (y2 - frame_height)
            y2 = frame_height

        return (
            max(0, x1),
            max(0, y1),
            min(frame_width, x2),
            min(frame_height, y2),
        )

    @staticmethod
    def _clamp_point(point: Tuple[float, float]) -> Tuple[float, float]:
        return (
            max(0.0, min(1.0, float(point[0]))),
            max(0.0, min(1.0, float(point[1]))),
        )
