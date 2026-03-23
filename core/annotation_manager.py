from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime

from loguru import logger


class AnnotationManager:
    """Manages rally annotations and YOLO bounding boxes."""

    def __init__(self, box_classes: Optional[Dict[int, str]] = None):
        self.box_classes: Dict[int, str] = dict(box_classes or {})
        self.rallies: List[Dict] = []
        self.actions: List[Dict] = []  # Derived from boxes inside rallies
        self.current_rally_start: Optional[Dict] = None
        self.yolo_boxes: Dict[int, Dict[int, Tuple]] = {}
        self.next_rally_id = 1
        self.next_action_id = 1
        self.next_box_id = 1

    def set_box_classes(self, box_classes: Dict[int, str]):
        """Update class map used for derived actions."""
        self.box_classes = dict(box_classes)
        self._rebuild_actions()

    def start_rally(self, frame_idx: int, fps: float = 30.0) -> bool:
        """Start a new rally annotation."""
        if self.current_rally_start is not None:
            logger.warning(
                f"Previous rally not completed (started at frame {self.current_rally_start['frame']})"
            )
            return False

        self.current_rally_start = {
            "frame": frame_idx,
            "fps": fps,
            "created_at": datetime.now().isoformat(),
        }
        logger.info(f"Rally started at frame {frame_idx}")
        return True

    def end_rally(self, frame_idx: int) -> Optional[Dict]:
        """Complete the current rally annotation."""
        if self.current_rally_start is None:
            logger.warning("No rally in progress to end")
            return None

        start_frame = self.current_rally_start["frame"]
        if frame_idx <= start_frame:
            logger.error(f"End frame {frame_idx} must be after start frame {start_frame}")
            return None

        fps = self.current_rally_start.get("fps", 30.0)
        rally = {
            "id": self.next_rally_id,
            "start_frame": start_frame,
            "end_frame": frame_idx,
            "type": "Rally",
            "start_time": start_frame / fps,
            "end_time": frame_idx / fps,
            "created_at": datetime.now().isoformat(),
            "actions": [],
            "hierarchy": [],
        }
        self.rallies.append(rally)
        self.next_rally_id += 1
        self.current_rally_start = None
        self._rebuild_actions()

        logger.info(
            f"Rally completed | Frames {start_frame}-{frame_idx} | "
            f"Duration: {rally['end_time'] - rally['start_time']:.2f}s"
        )
        return rally

    def cancel_rally(self) -> bool:
        """Cancel the current pending rally."""
        if self.current_rally_start is None:
            return False

        logger.info(f"Rally cancelled at frame {self.current_rally_start['frame']}")
        self.current_rally_start = None
        return True

    def delete_rally(self, rally_id: int) -> bool:
        """Delete a rally by ID."""
        for index, rally in enumerate(self.rallies):
            if rally.get("id") == rally_id:
                self.rallies.pop(index)
                self._rebuild_actions()
                logger.info(f"Rally deleted: ID {rally_id}")
                return True

        logger.warning(f"Rally not found: ID {rally_id}")
        return False

    def split_rally(self, frame_idx: int) -> bool:
        """Split the rally containing frame_idx into two rallies."""
        for index, rally in enumerate(self.rallies):
            start_frame = int(rally.get("start_frame", -1))
            end_frame = int(rally.get("end_frame", -1))
            if not (start_frame < frame_idx < end_frame):
                continue

            fps = self._fps_for_item(rally)
            first_rally = {
                "id": self.next_rally_id,
                "start_frame": start_frame,
                "end_frame": frame_idx,
                "type": "Rally",
                "start_time": start_frame / fps,
                "end_time": frame_idx / fps,
                "created_at": datetime.now().isoformat(),
                "actions": [],
                "hierarchy": [],
            }
            self.next_rally_id += 1

            second_rally = {
                "id": self.next_rally_id,
                "start_frame": frame_idx,
                "end_frame": end_frame,
                "type": "Rally",
                "start_time": frame_idx / fps,
                "end_time": end_frame / fps,
                "created_at": datetime.now().isoformat(),
                "actions": [],
                "hierarchy": [],
            }
            self.next_rally_id += 1

            self.rallies[index:index + 1] = [first_rally, second_rally]
            self._rebuild_actions()
            logger.info(f"Rally split at frame {frame_idx}: {start_frame}-{frame_idx} and {frame_idx}-{end_frame}")
            return True

        logger.warning(f"No rally found to split at frame {frame_idx}")
        return False

    def add_yolo_box(self, frame_idx: int, box: Tuple, validate: bool = True) -> int:
        """Add a YOLO bounding box to a frame."""
        cls_id, x, y, w, h = box[:5]

        if validate:
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

        if frame_idx not in self.yolo_boxes:
            self.yolo_boxes[frame_idx] = {}

        box_id = self._get_unique_box_id(frame_idx)
        self.yolo_boxes[frame_idx][box_id] = (cls_id, x, y, w, h)
        self._rebuild_actions()
        logger.debug(f"Box added to frame {frame_idx}: class={cls_id}, id={box_id}")
        return box_id

    def remove_yolo_box(self, frame_idx: int, box_index: int) -> bool:
        """Remove a specific bounding box from a frame by legacy index."""
        return self.remove_yolo_box_by_id(frame_idx, box_index)

    def remove_yolo_box_by_id(self, frame_idx: int, box_id: int) -> bool:
        """Remove a specific bounding box from a frame by box ID."""
        if frame_idx not in self.yolo_boxes:
            return False

        if box_id in self.yolo_boxes[frame_idx]:
            del self.yolo_boxes[frame_idx][box_id]
            if not self.yolo_boxes[frame_idx]:
                del self.yolo_boxes[frame_idx]
            self._rebuild_actions()
            logger.debug(f"Box removed from frame {frame_idx} by id {box_id}")
            return True

        logger.warning(f"Box with id {box_id} not found in frame {frame_idx}")
        return False

    def clear_frame_boxes(self, frame_idx: int):
        """Clear all boxes from a specific frame."""
        if frame_idx in self.yolo_boxes:
            del self.yolo_boxes[frame_idx]
            self._rebuild_actions()
            logger.debug(f"All boxes cleared from frame {frame_idx}")

    def export_yolo(self, output_dir: str) -> int:
        """Export YOLO format annotations."""
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        file_count = 0
        for frame_idx, boxes in self.yolo_boxes.items():
            if not boxes:
                continue

            path = os.path.join(labels_dir, f"{frame_idx:06d}.txt")
            with open(path, "w", encoding="utf-8") as file_obj:
                for _, box_data in boxes.items():
                    file_obj.write(" ".join(map(str, box_data)) + "\n")
            file_count += 1

        logger.info(f"YOLO export completed: {file_count} label files created in {labels_dir}")
        return file_count

    def export_actions(self, path: str) -> int:
        """Export rallies and derived actions to JSON file."""
        export_data = {
            "rallies": self.rallies,
            "actions": self.actions,
        }

        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(export_data, file_obj, indent=2, ensure_ascii=False)

        logger.info(f"Rallies exported: {len(self.rallies)} rallies saved to {path}")
        return len(self.rallies)

    def get_timeline_items(self) -> List[Dict]:
        """Return rally and derived action items for the timeline."""
        items = [dict(rally) for rally in self.rallies]
        items.extend(dict(action) for action in self.actions)
        return sorted(
            items,
            key=lambda item: (item.get("start_frame", 0), 0 if item.get("type") == "Rally" else 1),
        )

    def get_statistics(self) -> Dict:
        """Get annotation statistics."""
        action_counts: Dict[str, int] = {}
        for action in self.actions:
            action_type = action["type"]
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        total_frames_with_boxes = len(self.yolo_boxes)
        total_boxes = sum(len(box_dict) for box_dict in self.yolo_boxes.values())

        return {
            "total_rallies": len(self.rallies),
            "total_actions": len(self.actions),
            "action_counts": action_counts,
            "total_frames_with_boxes": total_frames_with_boxes,
            "total_boxes": total_boxes,
            "pending_rally": self.current_rally_start is not None,
        }

    def export_state(self) -> Dict:
        """Export full annotation state for project autosave."""
        return {
            "rallies": self.rallies,
            "actions": self.actions,
            "current_rally_start": self.current_rally_start,
            "yolo_boxes": {
                str(frame_idx): {
                    str(box_id): list(box_data) for box_id, box_data in boxes.items()
                }
                for frame_idx, boxes in self.yolo_boxes.items()
            },
            "next_rally_id": self.next_rally_id,
            "next_action_id": self.next_action_id,
            "next_box_id": self.next_box_id,
            "box_classes": self.box_classes,
        }

    def load_state(self, data: Dict):
        """Load annotation state from project JSON."""
        self.rallies = list(data.get("rallies", []))
        self.actions = list(data.get("actions", []))
        self.current_rally_start = data.get("current_rally_start")
        raw_boxes = data.get("yolo_boxes", {})
        self.yolo_boxes = {
            int(frame_idx): {
                int(box_id): tuple(box_data)
                for box_id, box_data in boxes.items()
            }
            for frame_idx, boxes in raw_boxes.items()
        }
        self.next_rally_id = int(data.get("next_rally_id", self._next_id(self.rallies)))
        self.next_action_id = int(data.get("next_action_id", self._next_id(self.actions)))
        self.next_box_id = int(data.get("next_box_id", self._max_box_id() + 1))

        if data.get("box_classes"):
            self.box_classes = {
                int(class_id): class_name for class_id, class_name in data["box_classes"].items()
            }

        self._rebuild_actions()

    def load_yolo_boxes(self, raw_boxes: Dict):
        """Load initial YOLO boxes from external JSON."""
        self.yolo_boxes = {
            int(frame_idx): {
                int(box_id): tuple(box_data)
                for box_id, box_data in boxes.items()
            }
            for frame_idx, boxes in raw_boxes.items()
        }
        self.next_box_id = self._max_box_id() + 1
        self._rebuild_actions()

    def update_box_class(self, frame_idx: int, box_id: int, new_class_id: int) -> bool:
        """Update class for an existing box."""
        if frame_idx not in self.yolo_boxes or box_id not in self.yolo_boxes[frame_idx]:
            return False

        old_box = self.yolo_boxes[frame_idx][box_id]
        self.yolo_boxes[frame_idx][box_id] = (new_class_id, old_box[1], old_box[2], old_box[3], old_box[4])
        self._rebuild_actions()
        return True

    def update_box_geometry(
        self,
        frame_idx: int,
        box_id: int,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> bool:
        """Update full geometry for an existing box."""
        if frame_idx not in self.yolo_boxes or box_id not in self.yolo_boxes[frame_idx]:
            return False

        self.yolo_boxes[frame_idx][box_id] = (
            class_id,
            max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, width)),
            max(0.0, min(1.0, height)),
        )
        self._rebuild_actions()
        return True

    def _get_unique_box_id(self, frame_idx: int) -> int:
        """Generate a unique box ID."""
        while True:
            candidate_id = self.next_box_id
            self.next_box_id += 1
            if frame_idx not in self.yolo_boxes or candidate_id not in self.yolo_boxes[frame_idx]:
                return candidate_id

    def _rebuild_actions(self):
        """Recompute derived actions from boxes for each rally."""
        rebuilt_actions: List[Dict] = []
        next_action_id = 1

        for rally in self.rallies:
            fps = self._fps_for_item(rally)
            rally_actions = self._derive_actions_for_range(rally["start_frame"], rally["end_frame"], fps)
            for action in rally_actions:
                action["id"] = next_action_id
                action["rally_id"] = rally["id"]
                next_action_id += 1
            rally["actions"] = rally_actions
            rally["hierarchy"] = self._build_rally_hierarchy(rally, rally_actions)
            rebuilt_actions.extend(rally_actions)

        self.actions = rebuilt_actions
        self.next_action_id = next_action_id

    def _derive_actions_for_range(self, start_frame: int, end_frame: int, fps: float) -> List[Dict]:
        """Build action ranges from boxes between start and end frames."""
        class_frames: Dict[str, Dict[str, int]] = {}

        for frame_idx in sorted(self.yolo_boxes):
            if frame_idx < start_frame or frame_idx > end_frame:
                continue

            for _, box_data in self.yolo_boxes[frame_idx].items():
                class_id = int(box_data[0])
                class_name = self.box_classes.get(class_id, f"Class {class_id}")
                if class_name.lower() == "rally":
                    continue

                if class_name not in class_frames:
                    class_frames[class_name] = {"start_frame": frame_idx, "end_frame": frame_idx}
                else:
                    class_frames[class_name]["start_frame"] = min(class_frames[class_name]["start_frame"], frame_idx)
                    class_frames[class_name]["end_frame"] = max(class_frames[class_name]["end_frame"], frame_idx)

        ordered_action_types = ["Serve", "Receive", "Set", "Attack"]
        ordered_names = sorted(
            class_frames,
            key=lambda name: (
                ordered_action_types.index(name) if name in ordered_action_types else len(ordered_action_types),
                class_frames[name]["start_frame"],
            ),
        )

        actions: List[Dict] = []
        for action_type in ordered_names:
            action_range = class_frames[action_type]
            actions.append(
                {
                    "start_frame": action_range["start_frame"],
                    "end_frame": action_range["end_frame"],
                    "type": action_type,
                    "start_time": action_range["start_frame"] / fps,
                    "end_time": action_range["end_frame"] / fps,
                }
            )

        return actions

    def _build_rally_hierarchy(self, rally: Dict, rally_actions: List[Dict]) -> List[Dict]:
        """Build persisted hierarchical representation for one rally."""
        hierarchy = [
            {
                "kind": "marker",
                "label": "rally_start",
                "type": "RallyStart",
                "frame": rally["start_frame"],
                "time": rally["start_time"],
                "rally_id": rally["id"],
            }
        ]

        for action in rally_actions:
            hierarchy.append(
                {
                    "kind": "action",
                    "label": action["type"].lower(),
                    "type": action["type"],
                    "start_frame": action["start_frame"],
                    "end_frame": action["end_frame"],
                    "start_time": action["start_time"],
                    "end_time": action["end_time"],
                    "rally_id": rally["id"],
                    "action_id": action["id"],
                }
            )

        hierarchy.append(
            {
                "kind": "marker",
                "label": "rally_end",
                "type": "RallyEnd",
                "frame": rally["end_frame"],
                "time": rally["end_time"],
                "rally_id": rally["id"],
            }
        )
        return hierarchy

    @staticmethod
    def _fps_for_item(item: Dict) -> float:
        duration_frames = item["end_frame"] - item["start_frame"]
        duration_time = item["end_time"] - item["start_time"]
        if duration_frames > 0 and duration_time > 0:
            return duration_frames / duration_time
        return 30.0

    @staticmethod
    def _next_id(items: List[Dict]) -> int:
        if not items:
            return 1
        return max(int(item.get("id", 0)) for item in items) + 1

    def _max_box_id(self) -> int:
        max_box_id = 0
        for boxes in self.yolo_boxes.values():
            if boxes:
                max_box_id = max(max_box_id, max(boxes))
        return max_box_id
