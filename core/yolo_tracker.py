"""YOLO tracker integration for automated object detection."""
from typing import List, Tuple, Optional
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class YOLOTracker:
    """Wrapper for YOLO model to provide automated detection."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO tracker.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available - detection features disabled")
            return
            
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if YOLO model is loaded and ready."""
        return self.model is not None
    
    def detect(self, frame: np.ndarray, target_classes: Optional[List[int]] = None) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Run detection on a frame.
        
        Args:
            frame: Input frame (BGR format)
            target_classes: List of class IDs to detect, None for all
            
        Returns:
            List of detections: [(class_id, x_center, y_center, width, height, confidence), ...]
            Coordinates are normalized to [0, 1]
        """
        if not self.is_available():
            logger.warning("YOLO model not available for detection")
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            if not results or len(results) == 0:
                return []
            
            # Extract detections from first result
            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return []
            
            detections = []
            h, w = frame.shape[:2]
            
            for box in boxes:
                # Get box data
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Filter by confidence and class
                if conf < self.confidence_threshold:
                    continue
                if target_classes is not None and cls_id not in target_classes:
                    continue
                
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                detections.append((cls_id, x_center, y_center, width, height, conf))
            
            logger.debug(f"Detected {len(detections)} objects with confidence >= {self.confidence_threshold}")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray], target_classes: Optional[List[int]] = None) -> List[List[Tuple]]:
        """
        Run detection on multiple frames (batch processing).
        
        Args:
            frames: List of input frames
            target_classes: List of class IDs to detect, None for all
            
        Returns:
            List of detection lists, one per frame
        """
        if not self.is_available():
            logger.warning("YOLO model not available for batch detection")
            return [[] for _ in frames]
        
        results_list = []
        for i, frame in enumerate(frames):
            detections = self.detect(frame, target_classes)
            results_list.append(detections)
            logger.debug(f"Batch detection: frame {i+1}/{len(frames)}, {len(detections)} detections")
        
        return results_list
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.confidence_threshold}")
