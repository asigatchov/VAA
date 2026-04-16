import numpy as np

from core.annotation_manager import AnnotationManager
from core.assistant_annotator import AssistantAnnotator, Detection


class DummyProcessor:
    def __init__(self, total_frames=20, width=1920, height=1080, fps=25.0):
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self._frames = []
        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[0, 0, 0] = frame_idx % 255
            self._frames.append(frame)

    def get_frame(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        return self._frames[frame_idx].copy()


class StubAssistantAnnotator(AssistantAnnotator):
    def __init__(self, detections, **kwargs):
        super().__init__(**kwargs)
        self._detections = list(detections)

    def _predict(self, frame_bgr: np.ndarray, threshold: float, status_callback=None) -> list[Detection]:
        del frame_bgr, status_callback
        return [item for item in self._detections if item.confidence >= threshold]

    def _predict_in_crop(self, frame_bgr: np.ndarray, center_norm, threshold: float, status_callback=None) -> list[Detection]:
        del frame_bgr, center_norm, status_callback
        return [item for item in self._detections if item.confidence >= threshold]


class FrameAwareStubAssistantAnnotator(AssistantAnnotator):
    def __init__(self, detections_by_frame, **kwargs):
        super().__init__(**kwargs)
        self._detections_by_frame = detections_by_frame

    def _predict(self, frame_bgr: np.ndarray, threshold: float, status_callback=None) -> list[Detection]:
        del status_callback
        frame_idx = int(frame_bgr[0, 0, 0])
        detections = self._detections_by_frame.get(frame_idx, [])
        return [item for item in detections if item.confidence >= threshold]

    def _predict_in_crop(self, frame_bgr: np.ndarray, center_norm, threshold: float, status_callback=None) -> list[Detection]:
        del center_norm, status_callback
        frame_idx = int(frame_bgr[0, 0, 0])
        detections = self._detections_by_frame.get(frame_idx, [])
        return [item for item in detections if item.confidence >= threshold]


def test_assistant_annotator_creates_nine_frame_clip_and_rally():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    annotator = StubAssistantAnnotator(
        detections=[
            Detection("person", 0.95, (900.0, 300.0, 1020.0, 840.0)),
            Detection("sports ball", 0.71, (1048.0, 516.0, 1064.0, 532.0)),
        ]
    )
    processor = DummyProcessor(total_frames=20)

    result = annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.5, 0.5),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )

    assert result["ok"] is True
    assert result["start_frame"] == 6
    assert result["end_frame"] == 14
    assert result["frames"] == 9
    assert result["boxes_created"] == 21
    assert len(result["ball_points"]) == 9
    assert result["model_variant"] == "medium"
    assert len(manager.rallies) == 1
    assert manager.rallies[0]["start_frame"] == 6
    assert manager.rallies[0]["end_frame"] == 14
    assert len(manager.actions) == 1
    assert manager.actions[0]["type"] == "Set"

    for frame_idx in range(6, 15):
        frame_boxes = manager.yolo_boxes[frame_idx]
        classes = sorted(int(box_data[0]) for box_data in frame_boxes.values())
        if 9 <= frame_idx <= 11:
            assert classes == [2, 5, 6]
        else:
            assert classes == [5, 6]


def test_assistant_annotator_replaces_existing_clip_classes_without_duplicates():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    annotator = StubAssistantAnnotator(
        detections=[
            Detection("person", 0.95, (900.0, 300.0, 1020.0, 840.0)),
            Detection("sports ball", 0.71, (1048.0, 516.0, 1064.0, 532.0)),
        ]
    )
    processor = DummyProcessor(total_frames=20)

    annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.5, 0.5),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )
    annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.52, 0.48),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )

    assert len(manager.rallies) == 1
    for frame_idx in range(6, 15):
        frame_boxes = manager.yolo_boxes[frame_idx]
        classes = sorted(int(box_data[0]) for box_data in frame_boxes.values())
        if 9 <= frame_idx <= 11:
            assert classes == [2, 5, 6]
        else:
            assert classes == [5, 6]


def test_assistant_annotator_returns_error_when_no_player_detected():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    annotator = StubAssistantAnnotator(
        detections=[
            Detection("sports ball", 0.71, (1048.0, 516.0, 1064.0, 532.0)),
        ]
    )
    processor = DummyProcessor(total_frames=20)

    result = annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.5, 0.5),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )

    assert result == {"ok": False, "reason": "no_player_detection"}


def test_assistant_prefers_person_closest_to_click():
    annotator = StubAssistantAnnotator(
        detections=[
            Detection("person", 0.99, (200.0, 100.0, 260.0, 500.0)),
            Detection("person", 0.80, (880.0, 200.0, 1040.0, 900.0)),
        ]
    )
    processor = DummyProcessor(total_frames=1)
    frame = processor.get_frame(0)

    picked = annotator._pick_player_detection(frame, (0.5, 0.5))

    assert picked is not None
    assert picked.xyxy == (880.0, 200.0, 1040.0, 900.0)


def test_assistant_uses_highest_confidence_ball_on_each_frame():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    processor = DummyProcessor(total_frames=20)
    annotator = FrameAwareStubAssistantAnnotator(
        detections_by_frame={
            frame_idx: [
                Detection("person", 0.90, (900.0, 300.0, 1020.0, 840.0)),
                Detection("sports ball", 0.20, (1000.0, 520.0, 1010.0, 530.0)),
                Detection("sports ball", 0.80, (1100.0, 540.0, 1112.0, 552.0)),
            ]
            for frame_idx in range(6, 15)
        }
    )

    result = annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.5, 0.5),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )

    assert result["ok"] is True
    expected_x = ((1100.0 + 1112.0) / 2.0) / processor.width
    expected_y = ((540.0 + 552.0) / 2.0) / processor.height
    expected_w = (1112.0 - 1100.0) / processor.width
    expected_h = (552.0 - 540.0) / processor.height

    for frame_idx in range(6, 15):
        frame_boxes = manager.yolo_boxes[frame_idx]
        ball_boxes = [box for box in frame_boxes.values() if int(box[0]) == 6]
        assert len(ball_boxes) == 1
        _, x_center, y_center, width, height = ball_boxes[0]
        assert abs(x_center - expected_x) < 1e-6
        assert abs(y_center - expected_y) < 1e-6
        assert abs(width - expected_w) < 1e-6
        assert abs(height - expected_h) < 1e-6


def test_action_box_covers_player_and_ball_only_on_center_triplet():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    processor = DummyProcessor(total_frames=20)
    annotator = FrameAwareStubAssistantAnnotator(
        detections_by_frame={
            frame_idx: [
                Detection("person", 0.90, (900.0, 300.0, 1020.0, 840.0)),
                Detection("sports ball", 0.80, (1100.0, 540.0, 1112.0, 552.0)),
            ]
            for frame_idx in range(6, 15)
        }
    )

    result = annotator.annotate_clip(
        processor=processor,
        annotations=manager,
        ball_lookup={},
        center_frame=10,
        click_center=(0.5, 0.5),
        action_class_id=2,
        player_class_id=5,
        ball_class_id=6,
    )

    assert result["ok"] is True

    for frame_idx in range(6, 15):
        frame_boxes = list(manager.yolo_boxes[frame_idx].values())
        action_boxes = [box for box in frame_boxes if int(box[0]) == 2]
        if frame_idx < 9 or frame_idx > 11:
            assert action_boxes == []
            continue

        assert len(action_boxes) == 1
        player_box = next(box for box in frame_boxes if int(box[0]) == 5)
        ball_box = next(box for box in frame_boxes if int(box[0]) == 6)
        action_box = action_boxes[0]

        def bounds(box):
            _, x, y, w, h = box
            return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

        p_x1, p_y1, p_x2, p_y2 = bounds(player_box)
        b_x1, b_y1, b_x2, b_y2 = bounds(ball_box)
        a_x1, a_y1, a_x2, a_y2 = bounds(action_box)

        assert a_x1 <= min(p_x1, b_x1) + 1e-6
        assert a_y1 <= min(p_y1, b_y1) + 1e-6
        assert a_x2 >= max(p_x2, b_x2) - 1e-6
        assert a_y2 >= max(p_y2, b_y2) - 1e-6
