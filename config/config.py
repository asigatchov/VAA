"""Configuration module for VAA application."""
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AnnotationConfig:
    """Configuration for annotation behavior and data processing."""
    
    # Video processing
    target_resolution: Tuple[int, int] = (1920, 1080)
    cache_limit_frames: int = 100
    default_fps: float = 30.0
    
    # YOLO integration
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.5
    auto_detect: bool = False
    
    # Export settings
    export_format: str = "yolo"
    
    # Action types (6 volleyball actions)
    action_types: Tuple[str, ...] = (
        "Serve", "Receive", "Set", "Attack"
    )
    
    # Box classes (action types as classes)
    box_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "Serve",
        1: "Receive",
        2: "Set",
        3: "Attack",
        5: "player",
        6: "ball",
    })


@dataclass
class UIConfig:
    """Configuration for UI appearance and behavior."""
    
    # Window settings
    window_size: Tuple[int, int] = (1600, 900)
    window_title: str = "Volleyball Action Annotator"
    
    # Display settings
    show_superframe_on_start: bool = False
    show_boxes_on_start: bool = True
    
    # Playback settings
    default_playback_speed: float = 1.0
    playback_speeds: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    
    # Box colors (class_id -> color) - matches action types
    box_colors: Dict[int, str] = field(default_factory=lambda: {
        0: "#FF6B6B",    # Serve
        1: "#4ECDC4",    # Receive
        2: "#45B7D1",    # Set
        3: "#FFA07A",    # Attack
        5: "#6C5CE7",    # player
        6: "#FFD60A",    # ball
    })
    selected_box_color: str = "#00FF00"
    box_line_width: int = 2
    selected_box_line_width: int = 4
    
    # Timeline settings
    timeline_height: int = 60
    action_marker_height: int = 20
    
    # Action type colors
    action_colors: Dict[str, str] = field(default_factory=lambda: {
        "Rally": "#B8C0FF",
        "Serve": "#FF6B6B",
        "Receive": "#4ECDC4",
        "Set": "#45B7D1",
        "Attack": "#FFA07A",
    })
    
    # UI language
    language: str = "en"
