"""Assistant clip annotation helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import rfdetr
from rfdetr.assets.coco_classes import COCO_CLASSES
from supervision import Detections as SupervisionDetections

from core.annotation_manager import AnnotationManager

try:
    import onnxruntime
except ImportError:  # pragma: no cover - exercised via runtime error path
    onnxruntime = None


PERSON_CLASS_NAME = "person"
BALL_CLASS_NAME = "sports ball"
MIXFORMER_BACKEND = "mixformer_v2_onnx"
MIXFORMER_MODEL_DEFAULTS = {
    "small": {"template_factor": 2.0, "search_factor": 4.5},
    "base": {"template_factor": 2.0, "search_factor": 4.5},
}


def _ensure_rfdetr_supervision_compat() -> None:
    """Provide the metadata API expected by newer RF-DETR on older supervision."""
    if hasattr(SupervisionDetections, "data"):
        return

    def _get_data(self) -> dict:
        data = self.__dict__.get("_compat_data")
        if data is None:
            data = {}
            self.__dict__["_compat_data"] = data
        return data

    SupervisionDetections.data = property(_get_data)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class PreprocessorXOnnx:
    """Prepare MixFormer image tensors for ONNX Runtime."""

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float32)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std
        amask_arr_3d = amask_arr[np.newaxis, :, :].astype(np.bool_)
        return img_arr_4d, amask_arr_3d


def sample_target(
    image: np.ndarray,
    target_bb: tuple[float, float, float, float],
    search_area_factor: float,
    output_sz: Optional[int] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Crop one square search region around the target."""
    x, y, w, h = [float(v) for v in target_bb]
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise ValueError("Too small bounding box.")

    x1 = int(round(x + 0.5 * w - 0.5 * crop_sz))
    y1 = int(round(y + 0.5 * h - 0.5 * crop_sz))
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y2_pad = max(y2 - image.shape[0] + 1, 0)

    image_crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    image_crop = cv2.copyMakeBorder(image_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    crop_h, crop_w = image_crop.shape[:2]
    att_mask = np.ones((crop_h, crop_w), dtype=np.float32)
    end_x = None if x2_pad == 0 else -x2_pad
    end_y = None if y2_pad == 0 else -y2_pad
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is None:
        return image_crop, 1.0, att_mask.astype(np.bool_)

    resize_factor = output_sz / crop_sz
    image_crop = cv2.resize(image_crop, (output_sz, output_sz))
    att_mask = cv2.resize(att_mask, (output_sz, output_sz), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    return image_crop, resize_factor, att_mask


def clip_box(box: list[float], height: int, width: int, margin: int = 0) -> list[float]:
    """Clamp one xywh box to the image area."""
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    x1 = min(max(0, x1), width - margin)
    x2 = min(max(margin, x2), width)
    y1 = min(max(0, y1), height - margin)
    y2 = min(max(margin, y2), height)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def infer_spatial_size(input_meta, fallback_name: str) -> int:
    """Read one static square input size from ONNX metadata."""
    shape = input_meta.shape
    if len(shape) != 4:
        raise ValueError(f"Unexpected input shape for {fallback_name}: {shape}")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(
            f"Dynamic spatial size for {fallback_name} is not supported; "
            "set template/search size explicitly"
        )
    if height != width:
        raise ValueError(f"Expected square spatial input for {fallback_name}, got {height}x{width}")
    return int(height)


def detect_mixformer_model_name(model_path: str) -> str:
    """Infer model family from filename for default search factors."""
    model_name = Path(model_path).name.lower()
    for candidate in MIXFORMER_MODEL_DEFAULTS:
        if candidate in model_name:
            return candidate
    return "small"


class MixFormerV2OnnxRunner:
    """Thin ONNX wrapper around one MixFormer tracker instance."""

    def __init__(self, session, template_size: int, search_size: int, template_factor: float, search_factor: float):
        self.session = session
        self.preprocessor = PreprocessorXOnnx()
        self.input_names = [item.name for item in session.get_inputs()]
        if len(self.input_names) != 3:
            raise ValueError(f"Expected 3 ONNX inputs, got {len(self.input_names)}")

        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.template_factor = float(template_factor)
        self.search_factor = float(search_factor)
        self.state: Optional[list[float]] = None
        self.template: Optional[np.ndarray] = None
        self.online_template: Optional[np.ndarray] = None

    def initialize(self, image_bgr: np.ndarray, init_bbox: tuple[float, float, float, float]) -> None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        z_patch_arr, _, z_amask_arr = sample_target(
            image_rgb,
            init_bbox,
            self.template_factor,
            output_sz=self.template_size,
        )
        template_input, _ = self.preprocessor.process(z_patch_arr, np.asarray(z_amask_arr))
        self.template = template_input
        self.online_template = template_input.copy()
        self.state = [float(value) for value in init_bbox]

    def track(self, image_bgr: np.ndarray) -> dict[str, object]:
        if self.state is None or self.template is None or self.online_template is None:
            raise RuntimeError("MixFormer runner used before initialize()")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image_rgb,
            tuple(self.state),
            self.search_factor,
            output_sz=self.search_size,
        )
        search_input, _ = self.preprocessor.process(x_patch_arr, np.asarray(x_amask_arr))

        ort_inputs = {
            self.input_names[0]: self.template.astype(np.float32),
            self.input_names[1]: self.online_template.astype(np.float32),
            self.input_names[2]: search_input.astype(np.float32),
        }
        pred_boxes, pred_scores = self.session.run(None, ort_inputs)

        pred_boxes = np.asarray(pred_boxes)
        if pred_boxes.ndim == 3:
            mean_box = pred_boxes.mean(axis=1).reshape(-1)
        elif pred_boxes.ndim == 2 and pred_boxes.shape[1] % 4 == 0:
            mean_box = pred_boxes.reshape(pred_boxes.shape[0], -1, 4).mean(axis=1).reshape(-1)
        else:
            mean_box = pred_boxes.reshape(-1)[:4]

        pred_box = (mean_box * self.search_size / resize_factor).tolist()
        pred_score = float(np.asarray(pred_scores).reshape(-1)[0])
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), height, width, margin=2)
        return {"target_bbox": tuple(self.state), "conf_score": pred_score}

    def _map_box_back(self, pred_box: list[float], resize_factor: float) -> list[float]:
        if self.state is None:
            raise RuntimeError("MixFormer runner used before initialize()")

        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]


@dataclass
class AssistantAnnotator:
    """Generate a short action clip with RF-DETR or MixFormerV2 ONNX."""

    clip_radius: int = 4
    crop_size_px: Tuple[int, int] = (960, 540)
    crop_mode: str = "full_frame"
    detector_backend: str = "rfdetr_medium"
    model_variant: str = "medium"
    resolution: int = 640
    person_threshold: float = 0.55
    ball_threshold: float = 0.12
    mixformer_model_path: str = "weights/mixformerv2_base.onnx"
    mixformer_template_size: Optional[int] = None
    mixformer_search_size: Optional[int] = None
    mixformer_template_factor: Optional[float] = None
    mixformer_search_factor: Optional[float] = None
    _model: object | None = field(default=None, init=False, repr=False)
    _model_backend: str | None = field(default=None, init=False, repr=False)
    _model_variant: str | None = field(default=None, init=False, repr=False)
    _model_resolution: int | None = field(default=None, init=False, repr=False)
    _mixformer_session: object | None = field(default=None, init=False, repr=False)
    _mixformer_session_model_path: str | None = field(default=None, init=False, repr=False)
    _mixformer_template_size_resolved: int | None = field(default=None, init=False, repr=False)
    _mixformer_search_size_resolved: int | None = field(default=None, init=False, repr=False)
    _mixformer_template_factor_resolved: float | None = field(default=None, init=False, repr=False)
    _mixformer_search_factor_resolved: float | None = field(default=None, init=False, repr=False)

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
        status_callback: Optional[Callable[[str], None]] = None,
        seed_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        seed_frame: Optional[int] = None,
    ) -> Dict[str, object]:
        """Annotate a short clip around one center frame."""
        del ball_lookup
        try:
            return self._annotate_clip_impl(
                processor=processor,
                annotations=annotations,
                center_frame=center_frame,
                click_center=click_center,
                action_class_id=action_class_id,
                player_class_id=player_class_id,
                ball_class_id=ball_class_id,
                replace_existing=replace_existing,
                progress_callback=progress_callback,
                status_callback=status_callback,
                seed_boxes=seed_boxes,
                seed_frame=seed_frame,
            )
        except RuntimeError as exc:
            return {"ok": False, "reason": "model_error", "message": str(exc)}
        except Exception as exc:
            return {"ok": False, "reason": "assistant_error", "message": str(exc)}

    def _annotate_clip_impl(
        self,
        processor,
        annotations: AnnotationManager,
        center_frame: int,
        click_center: Tuple[float, float],
        action_class_id: int,
        player_class_id: int,
        ball_class_id: int,
        replace_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        seed_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        seed_frame: Optional[int] = None,
    ) -> Dict[str, object]:
        """Internal annotate implementation."""
        total_frames = int(getattr(processor, "total_frames", 0))
        if total_frames <= 0:
            return {"ok": False, "reason": "no_frames"}

        center_image = processor.get_frame(int(center_frame))
        if center_image is None:
            return {"ok": False, "reason": "no_center_frame"}

        start_frame = max(0, int(center_frame) - self.clip_radius)
        end_frame = min(total_frames - 1, int(center_frame) + self.clip_radius)

        if self.detector_backend == MIXFORMER_BACKEND:
            mixformer_seed_frame = max(0, min(int(seed_frame if seed_frame is not None else start_frame), total_frames - 1))
            mixformer_end_frame = min(total_frames - 1, mixformer_seed_frame + self.clip_radius * 2)
            return self._annotate_clip_mixformer(
                processor=processor,
                annotations=annotations,
                center_frame=int(center_frame),
                start_frame=mixformer_seed_frame,
                end_frame=mixformer_end_frame,
                action_class_id=int(action_class_id),
                player_class_id=int(player_class_id),
                ball_class_id=int(ball_class_id),
                replace_existing=replace_existing,
                progress_callback=progress_callback,
                status_callback=status_callback,
                seed_boxes=seed_boxes,
            )

        return self._annotate_clip_rfdetr(
            processor=processor,
            annotations=annotations,
            center_frame=int(center_frame),
            center_image=center_image,
            click_center=click_center,
            start_frame=start_frame,
            end_frame=end_frame,
            action_class_id=int(action_class_id),
            player_class_id=int(player_class_id),
            ball_class_id=int(ball_class_id),
            replace_existing=replace_existing,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

    def _annotate_clip_rfdetr(
        self,
        processor,
        annotations: AnnotationManager,
        center_frame: int,
        center_image: np.ndarray,
        click_center: Tuple[float, float],
        start_frame: int,
        end_frame: int,
        action_class_id: int,
        player_class_id: int,
        ball_class_id: int,
        replace_existing: bool,
        progress_callback: Optional[Callable[[int, int, int], None]],
        status_callback: Optional[Callable[[str], None]],
    ) -> Dict[str, object]:
        """RF-DETR clip annotation path."""
        frame_height, frame_width = center_image.shape[:2]
        player_detection = self._pick_player_detection(center_image, click_center, status_callback=status_callback)
        if player_detection is None:
            return {"ok": False, "reason": "no_player_detection"}

        initial_player_center = self._center_from_xyxy(player_detection.xyxy, frame_width, frame_height)
        if replace_existing:
            annotations.clear_boxes_by_class_ids_in_range(
                start_frame,
                end_frame,
                {action_class_id, player_class_id, ball_class_id},
            )

        created_boxes = 0
        ball_points: Dict[int, Tuple[int, int]] = {}
        previous_player_center = initial_player_center
        action_start = max(start_frame, center_frame - 1)
        action_end = min(end_frame, center_frame + 1)
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
                status_callback=status_callback,
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
            player_box = self._make_box(player_class_id, player_center_frame, player_size)
            annotations.add_yolo_box(frame_idx, player_box)
            created_boxes += 1

            ball_box = None
            ball_detection_frame = self._pick_ball_detection_from_detections(frame_detections)
            if ball_detection_frame is not None:
                ball_center_frame = self._center_from_xyxy(ball_detection_frame.xyxy, frame_width, frame_height)
                ball_size = self._size_from_xyxy(ball_detection_frame.xyxy, frame_width, frame_height)
                ball_box = self._make_box(ball_class_id, ball_center_frame, ball_size)
                annotations.add_yolo_box(frame_idx, ball_box)
                created_boxes += 1
                ball_points[frame_idx] = self._normalized_to_source_point(
                    ball_center_frame,
                    processor,
                    frame_width,
                    frame_height,
                )

            if action_start <= frame_idx <= action_end:
                action_box = self._merge_boxes(action_class_id, player_box, ball_box)
                annotations.add_yolo_box(frame_idx, action_box)
                created_boxes += 1

        if progress_callback is not None:
            progress_callback(len(frame_indices), len(frame_indices), end_frame)

        return {
            "ok": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": end_frame - start_frame + 1,
            "boxes_created": created_boxes,
            "ball_points": ball_points,
            "created_rally": False,
            "model_variant": self.model_variant,
            "detector_backend": self.detector_backend,
        }

    def _annotate_clip_mixformer(
        self,
        processor,
        annotations: AnnotationManager,
        center_frame: int,
        start_frame: int,
        end_frame: int,
        action_class_id: int,
        player_class_id: int,
        ball_class_id: int,
        replace_existing: bool,
        progress_callback: Optional[Callable[[int, int, int], None]],
        status_callback: Optional[Callable[[str], None]],
        seed_boxes: Optional[Dict[str, Tuple[float, float, float, float]]],
    ) -> Dict[str, object]:
        """MixFormerV2 ONNX clip annotation path."""
        if not seed_boxes:
            return {"ok": False, "reason": "missing_seed_boxes", "message": "MixFormer requires player and ball seed boxes."}
        if "player" not in seed_boxes or "ball" not in seed_boxes:
            return {"ok": False, "reason": "missing_seed_boxes", "message": "MixFormer requires both player and ball seed boxes."}

        seed_image = processor.get_frame(start_frame)
        if seed_image is None:
            return {"ok": False, "reason": "no_seed_frame", "message": "Failed to read MixFormer seed frame."}

        frame_height, frame_width = seed_image.shape[:2]
        player_seed = self._normalized_box_to_xywh(seed_boxes["player"], frame_width, frame_height)
        ball_seed = self._normalized_box_to_xywh(seed_boxes["ball"], frame_width, frame_height)

        if replace_existing:
            annotations.clear_boxes_by_class_ids_in_range(
                start_frame,
                end_frame,
                {action_class_id, player_class_id, ball_class_id},
            )

        total_steps = max(1, (end_frame - start_frame + 1) * 2)
        if status_callback is not None:
            status_callback("MixFormerV2: tracking ball clip...")

        ball_tracks = self._track_mixformer_object(
            processor=processor,
            start_frame=start_frame,
            end_frame=end_frame,
            init_bbox=ball_seed,
            progress_offset=0,
            progress_total=total_steps,
            progress_callback=progress_callback,
            status_callback=status_callback,
            label="ball",
        )

        if status_callback is not None:
            status_callback("MixFormerV2: tracking player clip...")

        player_tracks = self._track_mixformer_object(
            processor=processor,
            start_frame=start_frame,
            end_frame=end_frame,
            init_bbox=player_seed,
            progress_offset=(end_frame - start_frame + 1),
            progress_total=total_steps,
            progress_callback=progress_callback,
            status_callback=status_callback,
            label="player",
        )

        created_boxes = 0
        ball_points: Dict[int, Tuple[int, int]] = {}
        frame_indices = list(range(start_frame, end_frame + 1))
        action_start = min(end_frame, start_frame + 3)
        action_end = min(end_frame, start_frame + 5)

        normalized_player_tracks = {
            frame_idx: self._xywh_to_normalized_box(player_class_id, xywh, frame_width, frame_height)
            for frame_idx, xywh in player_tracks.items()
        }
        normalized_ball_tracks = {
            frame_idx: self._xywh_to_normalized_box(ball_class_id, xywh, frame_width, frame_height)
            for frame_idx, xywh in ball_tracks.items()
        }

        for frame_idx in frame_indices:
            player_box = normalized_player_tracks.get(frame_idx)
            ball_box = normalized_ball_tracks.get(frame_idx)
            if player_box is not None:
                annotations.add_yolo_box(frame_idx, player_box)
                created_boxes += 1
            if ball_box is not None:
                annotations.add_yolo_box(frame_idx, ball_box)
                created_boxes += 1
                ball_points[frame_idx] = self._normalized_to_source_point(
                    (ball_box[1], ball_box[2]),
                    processor,
                    frame_width,
                    frame_height,
                )

            if action_start <= frame_idx <= action_end and player_box is not None:
                action_box = self._build_mixformer_action_box(
                    action_class_id,
                    frame_idx,
                    normalized_player_tracks,
                    normalized_ball_tracks,
                )
                annotations.add_yolo_box(frame_idx, action_box)
                created_boxes += 1

        if progress_callback is not None:
            progress_callback(total_steps, total_steps, end_frame)

        return {
            "ok": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": end_frame - start_frame + 1,
            "boxes_created": created_boxes,
            "ball_points": ball_points,
            "created_rally": False,
            "model_variant": self.model_variant,
            "detector_backend": self.detector_backend,
        }

    def _get_model(self, status_callback: Optional[Callable[[str], None]] = None):
        """Load the RF-DETR model lazily."""
        if (
            self._model is not None
            and self._model_backend == self.detector_backend
            and self._model_variant == self.model_variant
            and self._model_resolution == self.resolution
        ):
            return self._model

        self._model = None
        self._model_backend = None
        self._model_variant = None
        self._model_resolution = None

        if self.detector_backend != "rfdetr_medium":
            raise RuntimeError(f"Unsupported assistant detector backend: {self.detector_backend}")

        _ensure_rfdetr_supervision_compat()
        if status_callback is not None:
            status_callback("RF-DETR: loading model...")

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
        self._model_backend = self.detector_backend
        self._model_variant = self.model_variant
        self._model_resolution = self.resolution
        return self._model

    def _get_mixformer_session(self, status_callback: Optional[Callable[[str], None]] = None):
        """Load the MixFormer ONNX session lazily."""
        if onnxruntime is None:
            raise RuntimeError("MixFormer requires onnxruntime. Install project dependencies again.")

        model_path = Path(self.mixformer_model_path)
        if not model_path.is_file():
            raise RuntimeError(f"MixFormer model not found: {model_path}")

        if self._mixformer_session is not None and self._mixformer_session_model_path == str(model_path):
            return self._mixformer_session

        if status_callback is not None:
            status_callback(f"MixFormerV2: loading ONNX model {model_path.name}...")

        providers = onnxruntime.get_available_providers()
        self._mixformer_session = onnxruntime.InferenceSession(str(model_path), providers=providers)
        self._mixformer_session_model_path = str(model_path)

        inputs = self._mixformer_session.get_inputs()
        if len(inputs) != 3:
            raise RuntimeError(f"MixFormer ONNX must expose 3 inputs, got {len(inputs)}")

        model_name = detect_mixformer_model_name(str(model_path))
        self._mixformer_template_size_resolved = self.mixformer_template_size or infer_spatial_size(inputs[0], inputs[0].name)
        self._mixformer_search_size_resolved = self.mixformer_search_size or infer_spatial_size(inputs[2], inputs[2].name)
        defaults = MIXFORMER_MODEL_DEFAULTS.get(model_name, MIXFORMER_MODEL_DEFAULTS["small"])
        self._mixformer_template_factor_resolved = float(self.mixformer_template_factor or defaults["template_factor"])
        self._mixformer_search_factor_resolved = float(self.mixformer_search_factor or defaults["search_factor"])
        return self._mixformer_session

    def _create_mixformer_runner(self, status_callback: Optional[Callable[[str], None]] = None) -> MixFormerV2OnnxRunner:
        """Instantiate one fresh MixFormer tracker."""
        session = self._get_mixformer_session(status_callback=status_callback)
        return MixFormerV2OnnxRunner(
            session=session,
            template_size=int(self._mixformer_template_size_resolved or 112),
            search_size=int(self._mixformer_search_size_resolved or 224),
            template_factor=float(self._mixformer_template_factor_resolved or 2.0),
            search_factor=float(self._mixformer_search_factor_resolved or 4.5),
        )

    def _track_mixformer_object(
        self,
        processor,
        start_frame: int,
        end_frame: int,
        init_bbox: tuple[float, float, float, float],
        progress_offset: int,
        progress_total: int,
        progress_callback: Optional[Callable[[int, int, int], None]],
        status_callback: Optional[Callable[[str], None]],
        label: str,
    ) -> Dict[int, tuple[float, float, float, float]]:
        """Track one seeded object forward across the clip."""
        seed_image = processor.get_frame(start_frame)
        if seed_image is None:
            return {}

        tracks: Dict[int, tuple[float, float, float, float]] = {
            start_frame: tuple(float(value) for value in init_bbox)
        }
        step_index = progress_offset
        if progress_callback is not None:
            progress_callback(step_index, progress_total, start_frame)

        runner = self._create_mixformer_runner(status_callback=status_callback)
        runner.initialize(seed_image, init_bbox)
        for frame_idx in range(start_frame + 1, end_frame + 1):
            frame_image = processor.get_frame(frame_idx)
            if frame_image is None:
                continue
            result = runner.track(frame_image)
            tracks[frame_idx] = tuple(float(value) for value in result["target_bbox"])
            if progress_callback is not None:
                step_index += 1
                progress_callback(step_index, progress_total, frame_idx)
        if status_callback is not None:
            status_callback(f"MixFormerV2: {label} clip ready.")
        return tracks

    def _build_mixformer_action_box(
        self,
        class_id: int,
        frame_idx: int,
        player_tracks: Dict[int, Tuple[int, float, float, float, float]],
        ball_tracks: Dict[int, Tuple[int, float, float, float, float]],
    ) -> Tuple[int, float, float, float, float]:
        """Build one 3-frame action box from neighboring player and ball boxes."""
        player_box = player_tracks.get(frame_idx)
        if player_box is None:
            raise RuntimeError(f"Missing player box for action frame {frame_idx}")

        merged = (int(class_id), player_box[1], player_box[2], player_box[3], player_box[4])
        for neighbor_idx in range(frame_idx - 1, frame_idx + 2):
            neighbor_player = player_tracks.get(neighbor_idx)
            if neighbor_player is not None:
                merged = self._merge_boxes(class_id, merged, neighbor_player)
            neighbor_ball = ball_tracks.get(neighbor_idx)
            if neighbor_ball is not None:
                merged = self._merge_boxes(class_id, merged, neighbor_ball)
        return merged

    def _predict(
        self,
        frame_bgr: np.ndarray,
        threshold: float,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> list[Detection]:
        """Run RF-DETR on one frame."""
        model = self._get_model(status_callback=status_callback)

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
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> list[Detection]:
        """Run RF-DETR inside an optional crop around one point."""
        if self.crop_mode == "full_frame":
            return self._predict(frame_bgr, threshold=threshold, status_callback=status_callback)

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

        detections = self._predict(crop, threshold=threshold, status_callback=status_callback)
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
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Detection]:
        """Pick the nearest player to the user click."""
        frame_height, frame_width = frame_bgr.shape[:2]
        detections = self._predict_in_crop(
            frame_bgr,
            click_center,
            threshold=min(self.person_threshold, self.ball_threshold),
            status_callback=status_callback,
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
        """Pick the most plausible person around one reference point."""
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
        """Pick the highest-confidence ball candidate."""
        candidates = [
            detection
            for detection in detections
            if detection.class_name == BALL_CLASS_NAME and detection.confidence >= self.ball_threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    @staticmethod
    def _size_from_xyxy(
        xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        """Convert one xyxy box to normalized width and height."""
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
        """Convert one xyxy box to normalized center coordinates."""
        center_x, center_y = AssistantAnnotator._xyxy_center_px(xyxy)
        return AssistantAnnotator._clamp_point(
            (float(center_x) / max(1, frame_width), float(center_y) / max(1, frame_height))
        )

    @staticmethod
    def _xyxy_center_px(xyxy: tuple[float, float, float, float]) -> Tuple[float, float]:
        """Compute the center of one xyxy box in pixels."""
        x1, y1, x2, y2 = xyxy
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _normalized_to_source_point(
        center: Tuple[float, float],
        processor,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int]:
        """Project one normalized point from display space to source-video pixels."""
        source_width = max(1, int(getattr(processor, "width", frame_width)))
        source_height = max(1, int(getattr(processor, "height", frame_height)))
        return (
            int(round(float(center[0]) * source_width)),
            int(round(float(center[1]) * source_height)),
        )

    @staticmethod
    def _make_box(class_id: int, center: Tuple[float, float], size: Tuple[float, float]) -> Tuple[int, float, float, float, float]:
        """Create one normalized YOLO-style xywh box."""
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
        """Union two normalized boxes."""
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
        """Build a crop rectangle centered near one normalized point."""
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
        """Clamp one normalized point to the image area."""
        return (
            max(0.0, min(1.0, float(point[0]))),
            max(0.0, min(1.0, float(point[1]))),
        )

    @staticmethod
    def _normalized_box_to_xywh(
        normalized_box: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float, float]:
        """Convert normalized x_center,y_center,w,h to pixel xywh."""
        x_center, y_center, width, height = [float(value) for value in normalized_box]
        width_px = max(2.0, width * frame_width)
        height_px = max(2.0, height * frame_height)
        x1 = x_center * frame_width - width_px / 2.0
        y1 = y_center * frame_height - height_px / 2.0
        return tuple(clip_box([x1, y1, width_px, height_px], frame_height, frame_width, margin=2))

    @staticmethod
    def _xywh_to_normalized_box(
        class_id: int,
        xywh: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, float, float, float, float]:
        """Convert pixel xywh to normalized YOLO box."""
        x, y, w, h = [float(value) for value in xywh]
        x_center = (x + w / 2.0) / max(1, frame_width)
        y_center = (y + h / 2.0) / max(1, frame_height)
        width = w / max(1, frame_width)
        height = h / max(1, frame_height)
        return AssistantAnnotator._make_box(class_id, (x_center, y_center), (width, height))
