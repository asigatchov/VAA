"""Unit tests for AnnotationManager with box ID management."""
import pytest
import tempfile
import os
import json
from core.annotation_manager import AnnotationManager


class TestBoxIDManagement:
    """Test box ID assignment and uniqueness."""
    
    def test_add_box_assigns_unique_id(self):
        """Test that adding a box assigns a unique ID."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)  # cls, x, y, w, h
        
        box_id = manager.add_yolo_box(frame_idx=0, box=box)
        
        assert box_id == 1
        assert 0 in manager.yolo_boxes
        assert box_id in manager.yolo_boxes[0]
        assert manager.yolo_boxes[0][box_id] == box
    
    def test_multiple_boxes_get_different_ids(self):
        """Test that multiple boxes get different IDs."""
        manager = AnnotationManager()
        box1 = (0, 0.5, 0.5, 0.1, 0.1)
        box2 = (1, 0.3, 0.3, 0.1, 0.1)
        box3 = (0, 0.7, 0.7, 0.1, 0.1)
        
        id1 = manager.add_yolo_box(0, box1)
        id2 = manager.add_yolo_box(0, box2)
        id3 = manager.add_yolo_box(0, box3)
        
        assert id1 != id2 != id3
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        assert len(manager.yolo_boxes[0]) == 3
    
    def test_boxes_on_different_frames_can_have_same_id(self):
        """Test that boxes on different frames can have the same ID."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)
        
        id1 = manager.add_yolo_box(0, box)
        id2 = manager.add_yolo_box(1, box)
        
        # Different frames, IDs increment globally
        assert id1 == 1
        assert id2 == 2
    
    def test_unique_ids_per_frame(self):
        """Test that IDs are unique within a single frame."""
        manager = AnnotationManager()
        
        # Add 10 boxes to the same frame
        for i in range(10):
            box = (i % 2, 0.1 * i, 0.1 * i, 0.1, 0.1)
            manager.add_yolo_box(0, box)
        
        # All IDs in frame should be unique
        box_ids = set(manager.yolo_boxes[0].keys())
        assert len(box_ids) == 10
    
    def test_coordinate_validation(self):
        """Test that coordinates are clamped to [0, 1]."""
        manager = AnnotationManager()
        box = (0, 1.5, -0.5, 1.2, 0.8)  # Invalid coordinates
        
        box_id = manager.add_yolo_box(0, box, validate=True)
        
        stored_box = manager.yolo_boxes[0][box_id]
        cls_id, x, y, w, h = stored_box
        
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0


class TestBoxRemoval:
    """Test box removal by ID."""
    
    def test_remove_box_by_id(self):
        """Test removing a box by its ID."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)
        
        box_id = manager.add_yolo_box(0, box)
        assert box_id in manager.yolo_boxes[0]
        
        success = manager.remove_yolo_box_by_id(0, box_id)
        
        assert success is True
        # Frame should be removed when last box is deleted
        assert 0 not in manager.yolo_boxes
    
    def test_remove_nonexistent_box(self):
        """Test removing a box that doesn't exist."""
        manager = AnnotationManager()
        
        success = manager.remove_yolo_box_by_id(0, 999)
        
        assert success is False
    
    def test_remove_box_from_wrong_frame(self):
        """Test removing a box from the wrong frame."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)
        
        box_id = manager.add_yolo_box(0, box)
        success = manager.remove_yolo_box_by_id(1, box_id)  # Wrong frame
        
        assert success is False
        assert box_id in manager.yolo_boxes[0]  # Still exists
    
    def test_remove_last_box_clears_frame(self):
        """Test that removing the last box clears the frame entry."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)
        
        box_id = manager.add_yolo_box(0, box)
        manager.remove_yolo_box_by_id(0, box_id)
        
        assert 0 not in manager.yolo_boxes
    
    def test_remove_one_of_many_boxes(self):
        """Test removing one box from a frame with multiple boxes."""
        manager = AnnotationManager()
        
        id1 = manager.add_yolo_box(0, (0, 0.1, 0.1, 0.1, 0.1))
        id2 = manager.add_yolo_box(0, (1, 0.2, 0.2, 0.1, 0.1))
        id3 = manager.add_yolo_box(0, (0, 0.3, 0.3, 0.1, 0.1))
        
        manager.remove_yolo_box_by_id(0, id2)
        
        assert id2 not in manager.yolo_boxes[0]
        assert id1 in manager.yolo_boxes[0]
        assert id3 in manager.yolo_boxes[0]
        assert len(manager.yolo_boxes[0]) == 2
    
    def test_legacy_remove_by_index(self):
        """Test legacy remove_yolo_box method treats index as ID."""
        manager = AnnotationManager()
        box = (0, 0.5, 0.5, 0.1, 0.1)
        
        box_id = manager.add_yolo_box(0, box)
        success = manager.remove_yolo_box(0, box_id)
        
        assert success is True
        assert box_id not in manager.yolo_boxes.get(0, {})


class TestPasteWithNewIDs:
    """Test pasting boxes assigns new unique IDs."""
    
    def test_paste_creates_new_ids(self):
        """Test that pasting boxes creates new IDs."""
        manager = AnnotationManager()
        
        # Add original boxes to frame 0
        id1 = manager.add_yolo_box(0, (0, 0.1, 0.1, 0.1, 0.1))
        id2 = manager.add_yolo_box(0, (1, 0.2, 0.2, 0.1, 0.1))
        
        # Simulate copy-paste to frame 1
        original_boxes = list(manager.yolo_boxes[0].values())
        new_ids = []
        for box_data in original_boxes:
            new_id = manager.add_yolo_box(1, box_data)
            new_ids.append(new_id)
        
        # New IDs should be different
        assert id1 not in new_ids
        assert id2 not in new_ids
        assert len(set(new_ids)) == 2  # All unique
        assert len(manager.yolo_boxes[1]) == 2
    
    def test_paste_preserves_coordinates(self):
        """Test that pasting preserves box coordinates."""
        manager = AnnotationManager()
        
        original_box = (0, 0.123, 0.456, 0.789, 0.234)
        id1 = manager.add_yolo_box(0, original_box)
        
        # Copy to another frame
        box_data = manager.yolo_boxes[0][id1]
        id2 = manager.add_yolo_box(1, box_data)
        
        # Coordinates should match
        assert manager.yolo_boxes[0][id1] == manager.yolo_boxes[1][id2]
    
    def test_paste_multiple_times(self):
        """Test pasting the same boxes multiple times."""
        manager = AnnotationManager()
        
        # Original boxes
        manager.add_yolo_box(0, (0, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(0, (1, 0.2, 0.2, 0.1, 0.1))
        
        # Paste to frame 1
        for box_data in manager.yolo_boxes[0].values():
            manager.add_yolo_box(1, box_data)
        
        # Paste to frame 2
        for box_data in manager.yolo_boxes[0].values():
            manager.add_yolo_box(2, box_data)
        
        # Each frame should have 2 boxes with unique IDs
        assert len(manager.yolo_boxes[0]) == 2
        assert len(manager.yolo_boxes[1]) == 2
        assert len(manager.yolo_boxes[2]) == 2
        
        # All IDs should be unique globally
        all_ids = []
        for frame_boxes in manager.yolo_boxes.values():
            all_ids.extend(frame_boxes.keys())
        assert len(set(all_ids)) == 6


class TestRallyLifecycle:
    """Test rally start/end/cancel behavior."""

    def test_invalid_end_keeps_pending_rally(self):
        manager = AnnotationManager()

        assert manager.start_rally(100, fps=25.0) is True
        assert manager.end_rally(99) is None
        assert manager.current_rally_start is not None
        assert manager.current_rally_start["frame"] == 100
        assert manager.rallies == []

    def test_cancel_pending_rally(self):
        manager = AnnotationManager()

        assert manager.start_rally(100, fps=25.0) is True
        assert manager.cancel_rally() is True
        assert manager.current_rally_start is None
        assert manager.cancel_rally() is False

    def test_merge_rallies_creates_one_covering_range(self):
        manager = AnnotationManager({
            0: "Serve",
            1: "Receive",
            2: "Set",
            3: "Attack",
        })

        assert manager.start_rally(10, fps=25.0) is True
        manager.add_yolo_box(12, (0, 0.1, 0.1, 0.1, 0.1))
        assert manager.end_rally(20) is not None

        assert manager.start_rally(30, fps=25.0) is True
        manager.add_yolo_box(32, (1, 0.1, 0.1, 0.1, 0.1))
        assert manager.end_rally(40) is not None

        merged = manager.merge_rallies([1, 2])

        assert merged is not None
        assert len(manager.rallies) == 1
        assert manager.rallies[0]["id"] == merged["id"]
        assert manager.rallies[0]["start_frame"] == 10
        assert manager.rallies[0]["end_frame"] == 40
        assert [action["type"] for action in manager.rallies[0]["actions"]] == ["Serve", "Receive"]

    def test_merge_rallies_requires_two_existing_rallies(self):
        manager = AnnotationManager()

        assert manager.merge_rallies([1]) is None

    def test_clear_boxes_in_range(self):
        manager = AnnotationManager()

        manager.add_yolo_box(10, (0, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(11, (0, 0.2, 0.2, 0.1, 0.1))
        manager.add_yolo_box(20, (0, 0.3, 0.3, 0.1, 0.1))

        result = manager.clear_boxes_in_range(10, 11)

        assert result == {"frames_cleared": 2, "boxes_removed": 2}
        assert 10 not in manager.yolo_boxes
        assert 11 not in manager.yolo_boxes
        assert 20 in manager.yolo_boxes

    def test_rebuild_actions_keeps_repeated_volleyball_sequences(self):
        manager = AnnotationManager({
            0: "Serve",
            1: "Receive",
            2: "Set",
            3: "Attack",
        })

        assert manager.start_rally(100, fps=25.0) is True
        manager.add_yolo_box(100, (0, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(110, (1, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(120, (2, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(130, (3, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(140, (1, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(150, (2, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(160, (3, 0.1, 0.1, 0.1, 0.1))
        rally = manager.end_rally(170)

        assert rally is not None
        assert [action["type"] for action in manager.rallies[0]["actions"]] == [
            "Serve", "Receive", "Set", "Attack", "Receive", "Set", "Attack"
        ]
        assert [action["start_frame"] for action in manager.rallies[0]["actions"]] == [
            100, 110, 120, 130, 140, 150, 160
        ]

    def test_same_action_on_multiple_frames_stays_one_segment(self):
        manager = AnnotationManager({
            0: "Serve",
            1: "Receive",
            2: "Set",
            3: "Attack",
        })

        assert manager.start_rally(100, fps=25.0) is True
        manager.add_yolo_box(110, (1, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(115, (1, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(120, (2, 0.1, 0.1, 0.1, 0.1))
        rally = manager.end_rally(130)

        assert rally is not None
        actions = manager.rallies[0]["actions"]
        assert len(actions) == 2
        assert actions[0]["id"] == 1
        assert actions[0]["rally_id"] == 1
        assert actions[0]["start_frame"] == 110
        assert actions[0]["end_frame"] == 115
        assert actions[0]["type"] == "Receive"
        assert actions[0]["start_time"] == pytest.approx(110 / 25.0)
        assert actions[0]["end_time"] == pytest.approx(115 / 25.0)
        assert actions[1]["id"] == 2
        assert actions[1]["rally_id"] == 1
        assert actions[1]["start_frame"] == 120
        assert actions[1]["end_frame"] == 120
        assert actions[1]["type"] == "Set"
        assert actions[1]["start_time"] == pytest.approx(120 / 25.0)
        assert actions[1]["end_time"] == pytest.approx(120 / 25.0)

    def test_update_action_segment_class_updates_consecutive_frames(self):
        manager = AnnotationManager({
            0: "Serve",
            1: "Receive",
            2: "Set",
            3: "Attack",
        })

        assert manager.start_rally(100, fps=25.0) is True
        manager.add_yolo_box(120, (2, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(121, (2, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(122, (2, 0.1, 0.1, 0.1, 0.1))
        rally = manager.end_rally(130)

        assert rally is not None
        result = manager.update_action_segment_class(120, 122, "Set", 1)

        assert result == {"boxes_updated": 3, "frames_updated": 3}
        assert [manager.yolo_boxes[120][box_id][0] for box_id in manager.yolo_boxes[120]] == [1]
        assert [action["type"] for action in manager.rallies[0]["actions"]] == ["Receive"]
        assert manager.rallies[0]["actions"][0]["start_frame"] == 120
        assert manager.rallies[0]["actions"][0]["end_frame"] == 122


class TestYOLOExport:
    """Test YOLO format export."""
    
    def test_export_yolo_format(self):
        """Test exporting boxes to YOLO format."""
        manager = AnnotationManager()
        
        # Add boxes to different frames
        manager.add_yolo_box(0, (0, 0.5, 0.5, 0.1, 0.1))
        manager.add_yolo_box(0, (1, 0.3, 0.3, 0.1, 0.1))
        manager.add_yolo_box(1, (0, 0.7, 0.7, 0.1, 0.1))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_count = manager.export_yolo(tmpdir)
            
            assert file_count == 2  # 2 frames with boxes
            
            # Check frame 0
            frame0_path = os.path.join(tmpdir, "labels", "000000.txt")
            assert os.path.exists(frame0_path)
            with open(frame0_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # 2 boxes
                # Check format: class x y w h (no box_id)
                for line in lines:
                    parts = line.strip().split()
                    assert len(parts) == 5
            
            # Check frame 1
            frame1_path = os.path.join(tmpdir, "labels", "000001.txt")
            assert os.path.exists(frame1_path)
            with open(frame1_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1  # 1 box
    
    def test_export_skips_empty_frames(self):
        """Test that export skips frames with no boxes."""
        manager = AnnotationManager()
        
        manager.add_yolo_box(0, (0, 0.5, 0.5, 0.1, 0.1))
        # Frame 1 is empty
        manager.add_yolo_box(2, (0, 0.5, 0.5, 0.1, 0.1))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_count = manager.export_yolo(tmpdir)
            
            assert file_count == 2  # Only frames 0 and 2
            assert not os.path.exists(os.path.join(tmpdir, "labels", "000001.txt"))
    
    def test_export_coordinates_precision(self):
        """Test that exported coordinates maintain precision."""
        manager = AnnotationManager()
        
        box = (0, 0.123456, 0.789012, 0.345678, 0.901234)
        manager.add_yolo_box(0, box)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.export_yolo(tmpdir)
            
            frame0_path = os.path.join(tmpdir, "labels", "000000.txt")
            with open(frame0_path, "r") as f:
                line = f.readline().strip()
                parts = [float(x) for x in line.split()]
                
                # Check coordinates match
                assert parts[0] == 0  # class
                assert abs(parts[1] - 0.123456) < 1e-6
                assert abs(parts[2] - 0.789012) < 1e-6
                assert abs(parts[3] - 0.345678) < 1e-6
                assert abs(parts[4] - 0.901234) < 1e-6


class TestStatistics:
    """Test statistics calculation."""
    
    def test_statistics_with_boxes(self):
        """Test statistics with boxes."""
        manager = AnnotationManager()
        
        manager.add_yolo_box(0, (0, 0.5, 0.5, 0.1, 0.1))
        manager.add_yolo_box(0, (1, 0.3, 0.3, 0.1, 0.1))
        manager.add_yolo_box(1, (0, 0.7, 0.7, 0.1, 0.1))
        
        stats = manager.get_statistics()
        
        assert stats["total_frames_with_boxes"] == 2
        assert stats["total_boxes"] == 3
    
    def test_statistics_empty(self):
        """Test statistics with no boxes."""
        manager = AnnotationManager()
        
        stats = manager.get_statistics()
        
        assert stats["total_frames_with_boxes"] == 0
        assert stats["total_boxes"] == 0


class TestClearFrame:
    """Test clearing all boxes from a frame."""
    
    def test_clear_frame_boxes(self):
        """Test clearing all boxes from a frame."""
        manager = AnnotationManager()
        
        manager.add_yolo_box(0, (0, 0.1, 0.1, 0.1, 0.1))
        manager.add_yolo_box(0, (1, 0.2, 0.2, 0.1, 0.1))
        manager.add_yolo_box(1, (0, 0.3, 0.3, 0.1, 0.1))
        
        manager.clear_frame_boxes(0)
        
        assert 0 not in manager.yolo_boxes
        assert 1 in manager.yolo_boxes  # Frame 1 unaffected
    
    def test_clear_nonexistent_frame(self):
        """Test clearing a frame that doesn't exist."""
        manager = AnnotationManager()
        
        # Should not raise error
        manager.clear_frame_boxes(999)


def test_clear_boxes_by_class_ids_in_range_removes_only_matching_classes():
    manager = AnnotationManager({0: "Serve", 2: "Set", 5: "player", 6: "ball"})
    manager.add_yolo_box(10, (2, 0.5, 0.5, 0.1, 0.1))
    manager.add_yolo_box(10, (5, 0.5, 0.5, 0.1, 0.1))
    manager.add_yolo_box(10, (6, 0.5, 0.5, 0.02, 0.02))
    manager.add_yolo_box(11, (0, 0.5, 0.5, 0.1, 0.1))
    manager.add_yolo_box(11, (5, 0.5, 0.5, 0.1, 0.1))

    result = manager.clear_boxes_by_class_ids_in_range(10, 11, {5, 6})

    assert result == {"frames_cleared": 2, "boxes_removed": 3}
    frame10_classes = sorted(int(box[0]) for box in manager.yolo_boxes[10].values())
    frame11_classes = sorted(int(box[0]) for box in manager.yolo_boxes[11].values())
    assert frame10_classes == [2]
    assert frame11_classes == [0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
