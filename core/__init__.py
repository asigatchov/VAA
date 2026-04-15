"""Core components for VAA."""
from .video_processor import VideoProcessor
from .annotation_manager import AnnotationManager
from .assistant_annotator import AssistantAnnotator
from .yolo_tracker import YOLOTracker

__all__ = ['VideoProcessor', 'AnnotationManager', 'AssistantAnnotator', 'YOLOTracker']
