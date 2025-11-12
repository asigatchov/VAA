"""Core components for VAA."""
from .video_processor import VideoProcessor
from .annotation_manager import AnnotationManager
from .yolo_tracker import YOLOTracker

__all__ = ['VideoProcessor', 'AnnotationManager', 'YOLOTracker']
