from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime
from loguru import logger


class AnnotationManager:
    """Manages action annotations and YOLO bounding boxes."""
    
    def __init__(self):
        self.actions: List[Dict] = []  # List of action records
        self.current_action_start: Optional[Dict] = None  # Pending action (legacy)
        self.yolo_boxes: Dict[int, List[Tuple]] = {}  # {frame_idx: [(cls, x, y, w, h), ...]}
        self.next_action_id = 1  # Auto-incrementing action ID
        
        # Auto-tracking: active actions by type
        self.active_actions: Dict[str, Dict] = {}  # {action_type: {start_frame, start_time, fps}}

    def start_action(self, frame_idx: int, action_type: str, fps: float = 30.0) -> bool:
        """
        Start a new action annotation.
        
        Args:
            frame_idx: Starting frame index
            action_type: Type of action (Serve, Reception, etc.)
            fps: Video FPS for time calculation
            
        Returns:
            True if started successfully, False if action already pending
        """
        if self.current_action_start is not None:
            logger.warning(f"Previous action not completed (started at frame {self.current_action_start['frame']})")
            return False
            
        self.current_action_start = {
            "frame": frame_idx,
            "type": action_type,
            "fps": fps
        }
        logger.info(f"Action started: {action_type} at frame {frame_idx}")
        return True

    def end_action(self, frame_idx: int) -> Optional[Dict]:
        """
        Complete the current action annotation.
        
        Args:
            frame_idx: Ending frame index
            
        Returns:
            Completed action record or None if no action pending or invalid
        """
        if self.current_action_start is None:
            logger.warning("No action in progress to end")
            return None
            
        start_frame = self.current_action_start["frame"]
        if frame_idx <= start_frame:
            logger.error(f"End frame {frame_idx} must be after start frame {start_frame}")
            return None
            
        fps = self.current_action_start.get("fps", 30.0)
        action = {
            "id": self.next_action_id,
            "start_frame": start_frame,
            "end_frame": frame_idx,
            "type": self.current_action_start["type"],
            "start_time": start_frame / fps,
            "end_time": frame_idx / fps,
            "created_at": datetime.now().isoformat()
        }
        self.actions.append(action)
        self.next_action_id += 1
        self.current_action_start = None
        
        logger.info(
            f"Action completed: {action['type']} | "
            f"Frames {start_frame}-{frame_idx} | "
            f"Duration: {action['end_time'] - action['start_time']:.2f}s"
        )
        return action

    def cancel_action(self) -> bool:
        """
        Cancel the current pending action.
        
        Returns:
            True if action was cancelled, False if no action pending
        """
        if self.current_action_start is None:
            return False
        logger.info(f"Action cancelled: {self.current_action_start['type']} at frame {self.current_action_start['frame']}")
        self.current_action_start = None
        return True
    
    def delete_action(self, action_id: int) -> bool:
        """
        Delete an action by ID.
        
        Args:
            action_id: ID of action to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, action in enumerate(self.actions):
            if action.get("id") == action_id:
                self.actions.pop(i)
                logger.info(f"Action deleted: ID {action_id}")
                return True
        logger.warning(f"Action not found: ID {action_id}")
        return False
    
    def update_action(self, action_id: int, **kwargs) -> bool:
        """
        Update an action's properties.
        
        Args:
            action_id: ID of action to update
            **kwargs: Fields to update (type, start_frame, end_frame)
            
        Returns:
            True if updated, False if not found
        """
        for action in self.actions:
            if action.get("id") == action_id:
                # Update allowed fields
                if "type" in kwargs:
                    action["type"] = kwargs["type"]
                if "start_frame" in kwargs or "end_frame" in kwargs:
                    fps = action["end_time"] - action["start_time"]
                    fps = fps / (action["end_frame"] - action["start_frame"]) if fps > 0 else 30.0
                    if "start_frame" in kwargs:
                        action["start_frame"] = kwargs["start_frame"]
                        action["start_time"] = kwargs["start_frame"] / fps
                    if "end_frame" in kwargs:
                        action["end_frame"] = kwargs["end_frame"]
                        action["end_time"] = kwargs["end_frame"] / fps
                logger.info(f"Action updated: ID {action_id}")
                return True
        logger.warning(f"Action not found: ID {action_id}")
        return False

    def add_yolo_box(self, frame_idx: int, box: Tuple[int, float, float, float, float], validate: bool = True):
        """
        Add a YOLO bounding box to a frame.
        
        Args:
            frame_idx: Frame index
            box: Tuple of (class_id, x_center, y_center, width, height) in normalized coords [0-1]
            validate: Whether to validate and clamp coordinates
        """
        if validate:
            cls_id, x, y, w, h = box
            # Clamp coordinates to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            box = (cls_id, x, y, w, h)
            
        if frame_idx not in self.yolo_boxes:
            self.yolo_boxes[frame_idx] = []
        self.yolo_boxes[frame_idx].append(box)
        logger.debug(f"Box added to frame {frame_idx}: class={box[0]}")

    def remove_yolo_box(self, frame_idx: int, box_index: int) -> bool:
        """
        Remove a specific bounding box from a frame.
        
        Args:
            frame_idx: Frame index
            box_index: Index of box in the frame's box list
            
        Returns:
            True if removed, False if not found
        """
        if frame_idx in self.yolo_boxes and 0 <= box_index < len(self.yolo_boxes[frame_idx]):
            self.yolo_boxes[frame_idx].pop(box_index)
            if not self.yolo_boxes[frame_idx]:  # Remove frame entry if empty
                del self.yolo_boxes[frame_idx]
            logger.debug(f"Box removed from frame {frame_idx}, index {box_index}")
            return True
        return False
    
    def clear_frame_boxes(self, frame_idx: int):
        """Clear all boxes from a specific frame."""
        if frame_idx in self.yolo_boxes:
            del self.yolo_boxes[frame_idx]
            logger.debug(f"All boxes cleared from frame {frame_idx}")

    def export_yolo(self, output_dir: str) -> int:
        """
        Export YOLO format annotations.
        
        Args:
            output_dir: Directory to save label files
            
        Returns:
            Number of files created
        """
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        file_count = 0
        for frame_idx, boxes in self.yolo_boxes.items():
            if not boxes:  # Skip frames with no boxes
                continue
                
            path = os.path.join(labels_dir, f"{frame_idx:06d}.txt")
            with open(path, "w") as f:
                for box in boxes:
                    # Format: class_id x_center y_center width height
                    f.write(" ".join(map(str, box)) + "\n")
            file_count += 1
            
        logger.info(f"YOLO export completed: {file_count} label files created in {labels_dir}")
        return file_count

    def export_actions(self, path: str) -> int:
        """
        Export actions to JSON file.
        
        Args:
            path: Path to output JSON file
            
        Returns:
            Number of actions exported
        """
        # Prepare export data (exclude internal fields)
        export_data = []
        for action in self.actions:
            export_data.append({
                "start_time": action["start_time"],
                "end_time": action["end_time"],
                "type": action["type"],
                "start_frame": action["start_frame"],
                "end_frame": action["end_frame"]
            })
        
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Actions exported: {len(export_data)} actions saved to {path}")
        return len(export_data)

    def get_actions_at_frame(self, frame_idx: int) -> List[Dict]:
        """Get all actions that include the specified frame."""
        return [
            action for action in self.actions
            if action["start_frame"] <= frame_idx <= action["end_frame"]
        ]
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics."""
        action_counts = {}
        for action in self.actions:
            action_type = action["type"]
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        total_frames_with_boxes = len(self.yolo_boxes)
        total_boxes = sum(len(boxes) for boxes in self.yolo_boxes.values())
        
        return {
            "total_actions": len(self.actions),
            "action_counts": action_counts,
            "total_frames_with_boxes": total_frames_with_boxes,
            "total_boxes": total_boxes,
            "pending_action": self.current_action_start is not None,
            "active_actions": len(self.active_actions)
        }
    
    def auto_start_action(self, frame_idx: int, action_type: str, fps: float = 30.0):
        """Auto-start an action when a box appears (supports multiple simultaneous actions)."""
        if action_type not in self.active_actions:
            self.active_actions[action_type] = {
                "start_frame": frame_idx,
                "start_time": frame_idx / fps,
                "fps": fps,
                "last_frame": frame_idx
            }
            logger.info(f"Auto-started action: {action_type} at frame {frame_idx}")
    
    def auto_update_action(self, frame_idx: int, action_type: str):
        """Update last frame of an active action."""
        if action_type in self.active_actions:
            self.active_actions[action_type]["last_frame"] = frame_idx
    
    def auto_end_action(self, action_type: str) -> Optional[Dict]:
        """Auto-end an action when box disappears."""
        if action_type not in self.active_actions:
            return None
        
        active = self.active_actions.pop(action_type)
        start_frame = active["start_frame"]
        end_frame = active["last_frame"]
        fps = active.get("fps", 30.0)
        
        action = {
            "id": self.next_action_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "type": action_type,
            "start_time": start_frame / fps,
            "end_time": end_frame / fps,
            "created_at": datetime.now().isoformat(),
            "auto_tracked": True
        }
        self.actions.append(action)
        self.next_action_id += 1
        
        logger.info(
            f"Auto-completed action: {action_type} | "
            f"Frames {start_frame}-{end_frame} | "
            f"Duration: {action['end_time'] - action['start_time']:.2f}s"
        )
        return action
    
    def check_box_tracking(self, current_frame: int, fps: float = 30.0):
        """Check if boxes exist and manage auto-tracking."""
        # Get frames with boxes in a window around current frame
        window_start = max(0, current_frame - 1)
        window_end = current_frame + 1
        
        # Detect which action types have boxes in current frame
        current_box_types = set()
        if current_frame in self.yolo_boxes:
            # Map box class to action type (simplified: assume class 0 = ball detection)
            # For volleyball, we might detect ball presence
            current_box_types.add("Ball")  # Example: ball tracking
        
        # Check active actions
        for action_type in list(self.active_actions.keys()):
            if action_type in current_box_types:
                # Action still has boxes, update last frame
                self.auto_update_action(current_frame, action_type)
            else:
                # No boxes for this action type in current frame
                # Check if it's really disappeared (grace period)
                # For now, end it immediately
                pass  # Can add grace period logic here
