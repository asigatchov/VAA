"""Main window for VAA application."""
from typing import Optional
import csv
import json
from pathlib import Path
from datetime import datetime
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QStatusBar
)
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QImage, QPixmap

from core.video_processor import VideoProcessor
from core.annotation_manager import AnnotationManager
from core.yolo_tracker import YOLOTracker
from config import AnnotationConfig, UIConfig
from ui.image_canvas import ImageCanvas
from ui.widgets.timeline import TimelineWidget
from ui.widgets.action_panel import ActionPanel

from loguru import logger


class VideoAnnotationApp(QMainWindow):
    """Main application window for volleyball action annotation."""
    
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.annot_config = AnnotationConfig()
        self.ui_config = UIConfig()
        
        # Core components
        self.processor: Optional[VideoProcessor] = None
        self.annotations = AnnotationManager(self.annot_config.box_classes)
        self.yolo_tracker: Optional[YOLOTracker] = None
        self.projects_dir = Path(__file__).resolve().parents[1] / "projects"
        self.current_project_dir: Optional[Path] = None
        self.current_project_json: Optional[Path] = None
        self.current_video_path: Optional[Path] = None
        self.current_ball_csv_path: Optional[Path] = None
        self.current_action4_json_path: Optional[Path] = None
        self.current_ball_data = []
        self.current_ball_lookup = {}
        self.selected_rally_id: Optional[int] = None
        
        # State
        self.current_frame_idx = 0
        self.is_playing = False
        self.show_superframe = self.ui_config.show_superframe_on_start
        self.show_boxes = self.ui_config.show_boxes_on_start
        self.playback_speed = self.ui_config.default_playback_speed
        self.playback_step = 1
        
        # Setup UI
        self.setWindowTitle(self.ui_config.window_title)
        self.resize(*self.ui_config.window_size)
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_shortcuts()
        
        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.update_timer_interval()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        logger.info("Application initialized")
    
    def setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        # Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_box_colors(self.ui_config.box_colors)
        self.canvas.set_box_classes(self.annot_config.box_classes)
        self.canvas.box_added.connect(self.on_box_added)
        self.canvas.box_removed.connect(self.on_box_removed)
        self.canvas.box_class_changed.connect(self.on_box_class_changed)
        self.canvas.box_geometry_changed.connect(self.on_box_geometry_changed)
        self.canvas.ball_point_set.connect(self.on_ball_point_set)
        left_panel.addWidget(self.canvas, stretch=1)
        
        # Control bar
        control_bar = QHBoxLayout()
        
        self.jump_back_btn = QPushButton("<<")
        self.jump_back_btn.clicked.connect(lambda: self.step_frames(-15))
        self.jump_back_btn.setEnabled(False)
        control_bar.addWidget(self.jump_back_btn)

        self.prev_btn = QPushButton("<")
        self.prev_btn.clicked.connect(lambda: self.step_frames(-1))
        self.prev_btn.setEnabled(False)
        control_bar.addWidget(self.prev_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_bar.addWidget(self.play_btn)

        self.next_btn = QPushButton(">")
        self.next_btn.clicked.connect(lambda: self.step_frames(1))
        self.next_btn.setEnabled(False)
        control_bar.addWidget(self.next_btn)

        self.jump_forward_btn = QPushButton(">>")
        self.jump_forward_btn.clicked.connect(lambda: self.step_frames(15))
        self.jump_forward_btn.setEnabled(False)
        control_bar.addWidget(self.jump_forward_btn)

        self.rally_start_btn = QPushButton("[")
        self.rally_start_btn.clicked.connect(self.start_rally)
        self.rally_start_btn.setEnabled(False)
        control_bar.addWidget(self.rally_start_btn)

        self.rally_end_btn = QPushButton("]")
        self.rally_end_btn.clicked.connect(self.end_rally)
        self.rally_end_btn.setEnabled(False)
        control_bar.addWidget(self.rally_end_btn)

        self.rally_split_btn = QPushButton("|")
        self.rally_split_btn.clicked.connect(self.split_rally_at_current_frame)
        self.rally_split_btn.setEnabled(False)
        control_bar.addWidget(self.rally_split_btn)

        self.prev_annotated_btn = QPushButton("#")
        self.prev_annotated_btn.clicked.connect(self.prev_annotated_frame)
        self.prev_annotated_btn.setEnabled(False)
        control_bar.addWidget(self.prev_annotated_btn)

        self.next_annotated_btn = QPushButton("@")
        self.next_annotated_btn.clicked.connect(self.next_annotated_frame)
        self.next_annotated_btn.setEnabled(False)
        control_bar.addWidget(self.next_annotated_btn)
        
        # Add class selector for drawing
        control_bar.addWidget(QLabel("Draw Class:"))
        from PyQt6.QtWidgets import QComboBox
        self.class_combo = QComboBox()
        for class_id, class_name in sorted(self.annot_config.box_classes.items()):
            self.class_combo.addItem(class_name, class_id)
        self.class_combo.currentIndexChanged.connect(self.on_class_selection_changed)
        control_bar.addWidget(self.class_combo)
        
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        control_bar.addWidget(self.time_label)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        control_bar.addWidget(self.frame_label)
        
        control_bar.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        left_panel.addLayout(control_bar)
        
        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.set_action_colors(self.ui_config.action_colors)
        self.timeline.position_changed.connect(self.seek)
        left_panel.addWidget(self.timeline)

        self.detail_timeline = TimelineWidget(detail_mode=True)
        self.detail_timeline.set_action_colors(self.ui_config.action_colors)
        self.detail_timeline.position_changed.connect(self.seek)
        left_panel.addWidget(self.detail_timeline)
        
        main_layout.addLayout(left_panel, stretch=7)
        
        # Right panel - Action controls
        self.action_panel = ActionPanel(list(self.annot_config.action_types))
        self.action_panel.start_rally_requested.connect(self.start_rally)
        self.action_panel.end_rally_requested.connect(self.end_rally)
        self.action_panel.cancel_rally_requested.connect(self.cancel_rally)
        self.action_panel.action_type_selected.connect(self.on_action_type_selected)
        self.action_panel.action_reclass_requested.connect(self.on_action_reclass_requested)
        self.action_panel.rally_deleted.connect(self.delete_rally)
        self.action_panel.rally_selected.connect(self.on_rally_selected)
        self.action_panel.seek_to_rally.connect(self.seek)
        self.action_panel.playback_step_changed.connect(self.on_playback_step_changed)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.action_panel)
        
        # Export button
        self.export_btn = QPushButton("💾 Export Annotations")
        self.export_btn.clicked.connect(self.export_annotations)
        self.export_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.export_btn.setEnabled(False)
        right_panel.addWidget(self.export_btn)
        
        main_layout.addLayout(right_panel, stretch=3)
    
    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.load_video)
        file_menu.addAction(open_action)
        
        load_project_action = QAction("&Load Project...", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Annotation menu
        annot_menu = menubar.addMenu("&Annotation")

        toggle_superframe_action = QAction("Show &Superframe", self)
        toggle_superframe_action.setShortcut("F1")
        toggle_superframe_action.setCheckable(True)
        toggle_superframe_action.setChecked(self.show_superframe)
        toggle_superframe_action.triggered.connect(lambda checked: self.toggle_superframe(Qt.CheckState.Checked.value if checked else Qt.CheckState.Unchecked.value))
        annot_menu.addAction(toggle_superframe_action)
        
        toggle_boxes_action = QAction("Show &Bounding Boxes", self)
        toggle_boxes_action.setShortcut("F2")
        toggle_boxes_action.setCheckable(True)
        toggle_boxes_action.setChecked(self.show_boxes)
        toggle_boxes_action.triggered.connect(lambda checked: self.toggle_boxes(Qt.CheckState.Checked.value if checked else Qt.CheckState.Unchecked.value))
        annot_menu.addAction(toggle_boxes_action)

        toggle_ball_action = QAction("Show &Ball", self)
        toggle_ball_action.setCheckable(True)
        toggle_ball_action.setChecked(True)
        toggle_ball_action.triggered.connect(lambda checked: self.toggle_ball(Qt.CheckState.Checked.value if checked else Qt.CheckState.Unchecked.value))
        annot_menu.addAction(toggle_ball_action)

        mark_ball_action = QAction("&Mark Ball", self)
        mark_ball_action.setCheckable(True)
        mark_ball_action.triggered.connect(lambda checked: self.toggle_ball_markup(Qt.CheckState.Checked.value if checked else Qt.CheckState.Unchecked.value))
        annot_menu.addAction(mark_ball_action)
        
        start_action_act = QAction("&Start Rally", self)
        start_action_act.setShortcut("Return")
        start_action_act.triggered.connect(self.action_panel.start_btn.click)
        annot_menu.addAction(start_action_act)
        
        end_action_act = QAction("&End Rally", self)
        end_action_act.setShortcut("Shift+Return")
        end_action_act.triggered.connect(self.action_panel.end_btn.click)
        annot_menu.addAction(end_action_act)

        cancel_action_act = QAction("&Cancel Pending Rally", self)
        cancel_action_act.setShortcut("Escape")
        cancel_action_act.triggered.connect(self.cancel_rally)
        annot_menu.addAction(cancel_action_act)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Playback shortcuts
        play_shortcut = QAction(self)
        play_shortcut.setShortcut("Space")
        play_shortcut.triggered.connect(self.toggle_play)
        self.addAction(play_shortcut)
        
        prev_shortcut = QAction(self)
        prev_shortcut.setShortcut("Left")
        prev_shortcut.triggered.connect(lambda: self.step_frames(-1))
        self.addAction(prev_shortcut)
        
        next_shortcut = QAction(self)
        next_shortcut.setShortcut("Right")
        next_shortcut.triggered.connect(lambda: self.step_frames(1))
        self.addAction(next_shortcut)

        step_back_shortcut = QAction(self)
        step_back_shortcut.setShortcut("A")
        step_back_shortcut.triggered.connect(lambda: self.step_frames(-1))
        self.addAction(step_back_shortcut)

        step_forward_shortcut = QAction(self)
        step_forward_shortcut.setShortcut("D")
        step_forward_shortcut.triggered.connect(lambda: self.step_frames(1))
        self.addAction(step_forward_shortcut)

        jump_back_shortcut = QAction(self)
        jump_back_shortcut.setShortcut("S")
        jump_back_shortcut.triggered.connect(lambda: self.step_frames(-15))
        self.addAction(jump_back_shortcut)

        jump_forward_shortcut = QAction(self)
        jump_forward_shortcut.setShortcut("W")
        jump_forward_shortcut.triggered.connect(lambda: self.step_frames(15))
        self.addAction(jump_forward_shortcut)

        prev_annotated_shortcut = QAction(self)
        prev_annotated_shortcut.setShortcut("Q")
        prev_annotated_shortcut.triggered.connect(self.prev_annotated_frame)
        self.addAction(prev_annotated_shortcut)

        next_annotated_shortcut = QAction(self)
        next_annotated_shortcut.setShortcut("E")
        next_annotated_shortcut.triggered.connect(self.next_annotated_frame)
        self.addAction(next_annotated_shortcut)

        rally_start_shortcut = QAction(self)
        rally_start_shortcut.setShortcut("[")
        rally_start_shortcut.triggered.connect(self.start_rally)
        self.addAction(rally_start_shortcut)

        rally_end_shortcut = QAction(self)
        rally_end_shortcut.setShortcut("]")
        rally_end_shortcut.triggered.connect(self.end_rally)
        self.addAction(rally_end_shortcut)

        split_rally_shortcut = QAction(self)
        split_rally_shortcut.setShortcut("|")
        split_rally_shortcut.triggered.connect(self.split_rally_at_current_frame)
        self.addAction(split_rally_shortcut)
        
        home_shortcut = QAction(self)
        home_shortcut.setShortcut("Home")
        home_shortcut.triggered.connect(lambda: self.seek(0))
        self.addAction(home_shortcut)
        
        end_shortcut = QAction(self)
        end_shortcut.setShortcut("End")
        end_shortcut.triggered.connect(lambda: self.seek(self.processor.total_frames - 1) if self.processor else None)
        self.addAction(end_shortcut)
        
        # Copy/Paste shortcuts
        copy_shortcut = QAction(self)
        copy_shortcut.setShortcut("Ctrl+C")
        copy_shortcut.triggered.connect(self.copy_boxes)
        self.addAction(copy_shortcut)
        
        paste_shortcut = QAction(self)
        paste_shortcut.setShortcut("Ctrl+V")
        paste_shortcut.triggered.connect(self.paste_boxes)
        self.addAction(paste_shortcut)
    
    # Video loading and playback methods
    
    def load_video(self):
        """Load a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self._open_video_with_project(Path(file_path))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{str(e)}")
            logger.error(f"Video load failed: {e}")
    
    def toggle_play(self):
        """Toggle video playback."""
        if not self.processor:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.timer.start()
            self.play_btn.setText("Pause")
            self.status_bar.showMessage("Playing...")
        else:
            self.timer.stop()
            self.play_btn.setText("Play")
            self.status_bar.showMessage("Paused")
    
    def prev_frame(self):
        """Go to previous frame."""
        self.step_frames(-1)
    
    def next_frame(self):
        """Go to next frame."""
        if self.processor and self.current_frame_idx < self.processor.total_frames - 1:
            self.step_frames(self.playback_step)
        elif self.is_playing:
            # Stop playback at end
            self.toggle_play()

    def step_frames(self, offset: int):
        """Move current position by frame offset."""
        if not self.processor:
            return
        self.seek(self.current_frame_idx + offset)
    
    def seek(self, frame_idx: int):
        """Seek to specific frame."""
        if not self.processor:
            return
        
        self.current_frame_idx = max(0, min(frame_idx, self.processor.total_frames - 1))
        self.timeline.set_current_frame(self.current_frame_idx)
        self.detail_timeline.set_current_frame(self.current_frame_idx)
        self.update_display()

    def prev_annotated_frame(self):
        """Jump to previous frame that has box annotations."""
        if not self.processor or not self.annotations.yolo_boxes:
            return
        candidates = sorted(frame for frame, boxes in self.annotations.yolo_boxes.items() if boxes and frame < self.current_frame_idx)
        if candidates:
            self.seek(candidates[-1])
        else:
            self.status_bar.showMessage("No previous annotated frame")

    def next_annotated_frame(self):
        """Jump to next frame that has box annotations."""
        if not self.processor or not self.annotations.yolo_boxes:
            return
        candidates = sorted(frame for frame, boxes in self.annotations.yolo_boxes.items() if boxes and frame > self.current_frame_idx)
        if candidates:
            self.seek(candidates[0])
        else:
            self.status_bar.showMessage("No next annotated frame")
    
    def update_timer_interval(self):
        """Update timer interval based on playback speed."""
        if self.processor:
            interval = int(1000 / (self.processor.fps * self.playback_speed))
            self.timer.setInterval(max(1, interval))
    
    def toggle_superframe(self, state):
        """Toggle superframe display mode."""
        self.show_superframe = (state == Qt.CheckState.Checked.value)
        self.update_display()
    
    def toggle_boxes(self, state):
        """Toggle bounding box visibility."""
        self.show_boxes = (state == Qt.CheckState.Checked.value)
        self.canvas.toggle_boxes_visibility(self.show_boxes)

    def toggle_ball(self, state):
        """Toggle ball marker visibility."""
        self.canvas.toggle_ball_visibility(state == Qt.CheckState.Checked.value)

    def toggle_ball_markup(self, state):
        """Toggle manual ball markup mode."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.canvas.set_ball_markup_mode(enabled)
        self.status_bar.showMessage(
            "Ball markup mode enabled: click frame to set ball position" if enabled else "Ball markup mode disabled"
        )
    
   
    def on_class_selection_changed(self, index):
        class_id = self.class_combo.itemData(index)
        if class_id is not None:
            self.canvas.set_current_class(class_id)
            self.canvas.last_used_class_id = class_id
            class_name = self.annot_config.box_classes.get(class_id, f"Class {class_id}")
            self.status_bar.showMessage(f"Drawing class set to: {class_name}")

    def on_action_type_selected(self, action_type: str):
        """Sync action flow selection with draw class selector."""
        combo_index = self.class_combo.findText(action_type)
        if combo_index >= 0 and combo_index != self.class_combo.currentIndex():
            self.class_combo.setCurrentIndex(combo_index)
        self.status_bar.showMessage(f"Selected rally step: {action_type}")

    def on_playback_step_changed(self, step: int):
        """Update frame increment used during playback."""
        self.playback_step = max(1, min(30, int(step)))
        self.status_bar.showMessage(f"Playback step set to {self.playback_step} frame(s)")

    def on_rally_selected(self, rally_id: int):
        """Track selected rally for detail timeline."""
        self.selected_rally_id = rally_id if rally_id >= 0 else None
        self._update_detail_timeline()

    def on_action_reclass_requested(
        self,
        rally_id: int,
        start_frame: int,
        end_frame: int,
        old_action_type: str,
        new_action_type: str,
    ):
        """Apply a class change to all boxes inside one derived action segment."""
        del rally_id
        if old_action_type == new_action_type:
            self.status_bar.showMessage("Selected action already has that class")
            return

        new_class_id = next(
            (
                class_id for class_id, class_name in self.annot_config.box_classes.items()
                if class_name == new_action_type
            ),
            None,
        )
        if new_class_id is None:
            QMessageBox.warning(self, "Change Action Class", f"Class '{new_action_type}' is not configured.")
            return

        reply = QMessageBox.question(
            self,
            "Change Action Class",
            (
                f"Change action {old_action_type} {start_frame}-{end_frame} to {new_action_type}?\n\n"
                "All matching boxes on consecutive frames in this action segment will be updated."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        result = self.annotations.update_action_segment_class(
            start_frame,
            end_frame,
            old_action_type,
            new_class_id,
        )
        if result["boxes_updated"] <= 0:
            QMessageBox.information(self, "Change Action Class", "No matching boxes were found in this action segment.")
            return

        self._refresh_annotation_views()
        self._save_project_state()
        self.update_display()
        self.status_bar.showMessage(
            f"Changed action {old_action_type} {start_frame}-{end_frame} to {new_action_type}: "
            f"{result['boxes_updated']} boxes on {result['frames_updated']} frames"
        )

    def _pending_rally_preview(self):
        """Return temporary rally preview for an in-progress rally."""
        return self.annotations.get_pending_rally_preview(self.current_frame_idx)

    def _refresh_annotation_views(self):
        """Refresh rally list and both timelines from current annotation state."""
        self.action_panel.set_rallies(self.annotations.rallies, self._pending_rally_preview())
        pending_start = self.annotations.current_rally_start
        pending_frame = None if pending_start is None else int(pending_start.get("frame", 0))
        self.action_panel.set_pending_rally(pending_start is not None, pending_frame)
        focused_rally = self._focused_rally()
        focused_rally_id = None if focused_rally is None else int(focused_rally.get("id", -1))
        self.action_panel.set_current_context(self.current_frame_idx, focused_rally_id)
        self.timeline.set_actions(self.annotations.get_timeline_items())
        annotated_frames = self.annotations.get_annotated_timeline_frames()
        self.timeline.set_annotated_frames(annotated_frames)
        self.detail_timeline.set_annotated_frames(annotated_frames)
        self._update_detail_timeline()

    def update_display(self):
        """Update the canvas with current frame."""
        if not self.processor:
            return
        
        # Get frame (normal or superframe)
        if self.show_superframe:
            frame = self.processor.create_superframe(self.current_frame_idx)
        else:
            frame = self.processor.get_frame(self.current_frame_idx)
        
        if frame is None:
            logger.warning(f"Failed to get frame {self.current_frame_idx}")
            return
        
        # Convert BGR to RGB (OpenCV uses BGR, Qt uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QPixmap
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Get boxes for current frame
        boxes_dict = self.annotations.yolo_boxes.get(self.current_frame_idx, {})
        # Convert dict {box_id: (cls, x, y, w, h)} to list [(cls, x, y, w, h, box_id), ...]
        boxes = [(cls_id, x, y, w, h, box_id) for box_id, (cls_id, x, y, w, h) in boxes_dict.items()]
        
        # Update canvas
        self.canvas.set_image(pixmap, boxes, self._get_current_ball_position_normalized())
        
        # Update info labels
        self.update_info_labels()
        if self.annotations.current_rally_start is not None:
            self.action_panel.set_rallies(self.annotations.rallies, self._pending_rally_preview())
        focused_rally = self._focused_rally()
        focused_rally_id = None if focused_rally is None else int(focused_rally.get("id", -1))
        self.action_panel.set_current_context(self.current_frame_idx, focused_rally_id)
        self._update_detail_timeline()
    
    def update_info_labels(self):
        """Update time and frame information labels."""
        if not self.processor:
            return
        
        current_time = self.current_frame_idx / self.processor.fps
        total_time = self.processor.total_frames / self.processor.fps
        
        self.time_label.setText(
            f"{self.format_time(current_time)} / {self.format_time(total_time)}"
        )
        self.frame_label.setText(
            f"Frame: {self.current_frame_idx} / {self.processor.total_frames - 1}"
        )
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as MM:SS.d"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:04.1f}"
    
    # Annotation methods
    
    def start_rally(self):
        """Start a new rally annotation."""
        if not self.processor:
            return
        
        success = self.annotations.start_rally(
            self.current_frame_idx,
            self.processor.fps
        )
        
        if success:
            self._refresh_annotation_views()
            self._save_project_state()
            self.status_bar.showMessage(f"Rally started at frame {self.current_frame_idx}")
        else:
            self._refresh_annotation_views()
            start_frame = None
            if self.annotations.current_rally_start is not None:
                start_frame = int(self.annotations.current_rally_start["frame"])
            QMessageBox.warning(
                self,
                "Rally Already Started",
                (
                    f"A rally is already pending from frame {start_frame}. "
                    "End it, or cancel it with Escape before starting a new one."
                ) if start_frame is not None else
                "Please end the current rally before starting a new one."
            )
    
    def end_rally(self):
        """End the current rally annotation."""
        if not self.processor:
            return
        
        rally = self.annotations.end_rally(self.current_frame_idx)

        if rally:
            self._refresh_annotation_views()
            self._save_project_state()
            self.status_bar.showMessage(
                f"Rally completed: {rally['start_frame']} - {rally['end_frame']}"
            )
        else:
            pending_start = self.annotations.current_rally_start
            self._refresh_annotation_views()
            if pending_start is not None:
                start_frame = int(pending_start["frame"])
                QMessageBox.warning(
                    self,
                    "Invalid Rally End",
                    (
                        f"Current frame {self.current_frame_idx} is before the pending rally start "
                        f"at frame {start_frame}. Move forward, or cancel the pending rally with Escape."
                    )
                )
            else:
                QMessageBox.warning(
                    self,
                    "No Rally to End",
                    "Please start a rally first."
                )

    def cancel_rally(self):
        """Cancel the current pending rally."""
        if self.annotations.cancel_rally():
            self._refresh_annotation_views()
            self._save_project_state()
            self.status_bar.showMessage("Pending rally cancelled")
            return

        self._refresh_annotation_views()
        QMessageBox.information(
            self,
            "No Pending Rally",
            "There is no pending rally to cancel."
        )
    
    def delete_rally(self, rally_id: int):
        """Delete a rally."""
        rally = next((item for item in self.annotations.rallies if int(item.get("id", -1)) == int(rally_id)), None)
        if rally is None:
            QMessageBox.warning(
                self,
                "Delete Rally",
                "Selected rally was not found."
            )
            return

        action_count = len(rally.get("actions", []))
        start_frame = int(rally.get("start_frame", 0))
        end_frame = int(rally.get("end_frame", 0))
        action_label = "action" if action_count == 1 else "actions"
        box_summary = self._count_boxes_in_range(start_frame, end_frame)
        ball_count = self._count_ball_points_in_range(start_frame, end_frame)

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Delete Rally")
        dialog.setText(f"Choose how to delete rally {start_frame}-{end_frame}.")
        dialog.setInformativeText(
            f"Rally contains {action_count} derived {action_label}.\n"
            f"Range markup: {box_summary['boxes']} boxes on {box_summary['frames']} frames"
            f" and {ball_count} ball points."
        )
        only_rally_button = dialog.addButton("Only Rally", QMessageBox.ButtonRole.AcceptRole)
        with_markup_button = dialog.addButton("Rally + Markup", QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(only_rally_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked == only_rally_button:
            if self.annotations.delete_rally(rally_id):
                self._refresh_annotation_views()
                self._save_project_state()
                self.status_bar.showMessage(
                    f"Deleted rally {start_frame}-{end_frame}; frame markup kept"
                )
            return

        if clicked == with_markup_button:
            if self.annotations.delete_rally(rally_id):
                removed_boxes = self.annotations.clear_boxes_in_range(start_frame, end_frame)
                removed_ball_points = self._clear_ball_points_in_range(start_frame, end_frame)
                self._refresh_annotation_views()
                self._save_project_state()
                self.update_display()
                self.status_bar.showMessage(
                    f"Deleted rally {start_frame}-{end_frame} and markup: "
                    f"{removed_boxes['boxes_removed']} boxes, {removed_ball_points} ball points"
                )

    def split_rally_at_current_frame(self):
        """Split the rally containing the current frame into two rallies."""
        if not self.processor:
            return

        if self.annotations.split_rally(self.current_frame_idx):
            self._refresh_annotation_views()
            self._save_project_state()
            self.status_bar.showMessage(f"Rally split at frame {self.current_frame_idx}")
        else:
            QMessageBox.warning(
                self,
                "Split Rally",
                "Current frame must be inside an existing rally, not at its boundary."
            )
    
    
    def on_box_added(self, box):
        """Handle box added event from canvas."""
        box_id = self.annotations.add_yolo_box(self.current_frame_idx, box)
        self._refresh_annotation_views()
        self._save_project_state()
        self.status_bar.showMessage(f"Box #{box_id} added to frame {self.current_frame_idx}")
        logger.debug(f"Box added: {box}, assigned id={box_id}")
        self.update_display()
    
    def on_box_removed(self, box_id_or_index):
        """Handle box removed event from canvas.
        
        Args:
            box_id_or_index: Can be box_id (int) for deletion by ID, or index for legacy support
        """
        # Try to remove by ID first (new behavior)
        if self.annotations.remove_yolo_box_by_id(self.current_frame_idx, box_id_or_index):
            self.status_bar.showMessage(f"Box #{box_id_or_index} removed from frame {self.current_frame_idx}")
            logger.debug(f"Box removed by id: {box_id_or_index}")
        # Fallback to index-based removal (legacy behavior)
        elif self.annotations.remove_yolo_box(self.current_frame_idx, box_id_or_index):
            self.status_bar.showMessage(f"Box removed from frame {self.current_frame_idx}")
            logger.debug(f"Box removed by index: {box_id_or_index}")
        else:
            logger.warning(f"Failed to remove box {box_id_or_index} from frame {self.current_frame_idx}")
        self._refresh_annotation_views()
        self._save_project_state()
        self.update_display()



    def on_box_class_changed(self, box_id_or_index: int, new_class_id: int):
        """Handle box class change event from canvas."""
        updated_box_id = None
        updated = self.annotations.update_box_class(self.current_frame_idx, box_id_or_index, new_class_id)

        if updated:
            updated_box_id = box_id_or_index
        elif self.current_frame_idx in self.annotations.yolo_boxes:
            boxes_dict = self.annotations.yolo_boxes[self.current_frame_idx]
            box_ids = list(boxes_dict.keys())
            if 0 <= box_id_or_index < len(box_ids):
                updated_box_id = box_ids[box_id_or_index]
                updated = self.annotations.update_box_class(self.current_frame_idx, updated_box_id, new_class_id)

        if updated and updated_box_id is not None:
            class_name = self.annot_config.box_classes.get(new_class_id, f"Class {new_class_id}")
            self.status_bar.showMessage(f"Box #{updated_box_id} class changed to {class_name}")
            logger.info(f"Frame {self.current_frame_idx}, Box #{updated_box_id} class changed to {class_name}")
        else:
            logger.warning(
                f"Failed to change box class for {box_id_or_index} on frame {self.current_frame_idx}"
            )
        self._refresh_annotation_views()
        self._save_project_state()
        self.update_display()

    def on_box_geometry_changed(
        self,
        box_index: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        box_id: int,
    ):
        """Persist moved/resized box geometry from canvas back to annotation manager."""
        if self.current_frame_idx not in self.annotations.yolo_boxes:
            return

        boxes_dict = self.annotations.yolo_boxes[self.current_frame_idx]
        if box_id not in boxes_dict:
            return

        class_id = int(boxes_dict[box_id][0])
        updated = self.annotations.update_box_geometry(
            self.current_frame_idx,
            box_id,
            class_id,
            x_center,
            y_center,
            width,
            height,
        )
        if updated:
            self._refresh_annotation_views()
            self._save_project_state()

    def on_ball_point_set(self, x_norm: float, y_norm: float):
        """Persist clicked ball point in source-frame coordinates."""
        if not self.processor:
            return

        source_x = int(round(x_norm * self.processor.width))
        source_y = int(round(y_norm * self.processor.height))
        self._upsert_ball_point(self.current_frame_idx, source_x, source_y)
        self._save_project_state()
        self.status_bar.showMessage(f"Ball marked at frame {self.current_frame_idx}: ({source_x}, {source_y})")
        self.update_display()
        
    def copy_boxes(self):
        """Copy all boxes from current frame to clipboard (Ctrl+C)."""
        if not self.processor:
            return
        
        count = self.canvas.copy_boxes()
        if count > 0:
            self.status_bar.showMessage(f"Copied {count} boxes from frame {self.current_frame_idx}")
        else:
            self.status_bar.showMessage("No boxes to copy")
    
    def paste_boxes(self):
        """Paste boxes from clipboard to current frame (Ctrl+V)."""
        if not self.processor:
            return
        
        # Get clipboard boxes from canvas
        clipboard_boxes = self.canvas.clipboard_boxes
        if not clipboard_boxes:
            self.status_bar.showMessage("No boxes in clipboard to paste")
            return
        
        # Add boxes directly to annotation manager with new unique IDs
        pasted_ids = []
        for box in clipboard_boxes:
            # Strip old ID if present (will get new ID)
            if len(box) == 6:
                box_without_id = box[:5]
            else:
                box_without_id = box
            
            # Add box and get new unique ID
            new_id = self.annotations.add_yolo_box(self.current_frame_idx, box_without_id)
            pasted_ids.append(new_id)
        
        # Reload display to show pasted boxes
        self._refresh_annotation_views()
        self._save_project_state()
        self.update_display()
        self.status_bar.showMessage(f"Pasted {len(clipboard_boxes)} boxes to frame {self.current_frame_idx} (IDs: {', '.join(map(str, pasted_ids))})")
        logger.info(f"Pasted {len(clipboard_boxes)} boxes to frame {self.current_frame_idx} with new IDs: {pasted_ids}")
    
    # Export methods
    
    def export_annotations(self):
        """Export annotations to files."""
        if not self.processor:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        # Check if there are annotations
        stats = self.annotations.get_statistics()
        if stats["total_rallies"] == 0 and stats["total_boxes"] == 0:
            reply = QMessageBox.question(
                self,
                "No Annotations",
                "No annotations found. Export anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            str(self.current_project_dir) if self.current_project_dir else ""
        )
        
        if not output_dir:
            return
        
        try:
            # Export YOLO labels
            label_count = self.annotations.export_yolo(output_dir)
            
            # Export actions JSON
            import os
            actions_path = os.path.join(output_dir, "actions.json")
            action_count = self.annotations.export_actions(actions_path)
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Annotations exported successfully!\n\n"
                f"YOLO labels: {label_count} files\n"
                f"Rallies: {action_count} items\n\n"
                f"Location: {output_dir}"
            )
            
            self.status_bar.showMessage(f"Export complete: {output_dir}")
            logger.info(f"Export successful: {label_count} labels, {action_count} rallies")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export annotations:\n{str(e)}")
            logger.error(f"Export failed: {e}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About VAA",
            "<h2>Volleyball Action Annotator</h2>"
            "<p>Version 1.0</p>"
            "<p>A tool for annotating volleyball match videos with action boundaries and YOLO bounding boxes.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>3-frame RGB superframe visualization</li>"
            "<li>Action annotation with multiple types</li>"
            "<li>YOLO format bounding box export</li>"
            "<li>Timeline with action markers</li>"
            "</ul>"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Check for unsaved annotations
        if self.annotations.rallies or self.annotations.yolo_boxes:
            reply = QMessageBox.question(
                self,
                "Unsaved Annotations",
                "You have unsaved annotations. Do you want to exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # Cleanup
        if self.processor:
            self.processor.release()
        
        logger.info("Application closed")
        event.accept()

    def save_project(self):
        """Save current project JSON."""
        if not self.current_project_json:
            QMessageBox.warning(self, "No Project", "Please load a video or project first.")
            return

        self._save_project_state()
        self.status_bar.showMessage(f"Project saved: {self.current_project_json}")

    def load_project(self):
        """Load an existing project JSON."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            str(self.projects_dir),
            "Project Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        project_json = Path(file_path)
        with open(project_json, "r", encoding="utf-8") as file_obj:
            project_data = json.load(file_obj)

        video_path = Path(project_data["video_path"])
        if not video_path.exists():
            QMessageBox.critical(self, "Load Project Error", f"Video file not found:\n{video_path}")
            return

        self._open_video_with_project(video_path, project_json)

    def _open_video_with_project(self, video_path: Path, project_json: Optional[Path] = None):
        """Open a video file and optionally load a specific project JSON."""
        if self.processor:
            self.processor.release()

        self.processor = VideoProcessor(
            str(video_path),
            target_size=self.annot_config.target_resolution,
            cache_limit=self.annot_config.cache_limit_frames
        )

        self.current_frame_idx = 0
        self.is_playing = False
        self.annotations = AnnotationManager(self.annot_config.box_classes)
        self.current_ball_data = []
        self.current_ball_lookup = {}
        self.selected_rally_id = None

        if project_json is not None:
            self.current_project_dir = project_json.parent
            self.current_project_json = project_json
            self.current_video_path = video_path
            self._load_project_state(project_json)
        else:
            self._load_or_create_project(video_path)

        self.timeline.set_total_frames(self.processor.total_frames)
        self.timeline.set_current_frame(0)
        self.play_btn.setEnabled(True)
        self.jump_back_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.jump_forward_btn.setEnabled(True)
        self.rally_start_btn.setEnabled(True)
        self.rally_end_btn.setEnabled(True)
        self.rally_split_btn.setEnabled(True)
        self.prev_annotated_btn.setEnabled(True)
        self.next_annotated_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self._refresh_annotation_views()
        self.detail_timeline.set_total_frames(self.processor.total_frames)
        self.update_display()
        self.status_bar.showMessage(
            f"Loaded: {video_path} | {self.processor.total_frames} frames @ {self.processor.fps:.2f} fps"
        )

    def _load_or_create_project(self, video_path: Path):
        """Create or load a project for the opened video."""
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        project_name = video_path.stem
        project_dir = self.projects_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        project_json = project_dir / f"{project_name}.json"
        ball_csv_path = video_path.with_name(f"{video_path.stem}_predict_ball.csv")

        self.current_project_dir = project_dir
        self.current_project_json = project_json
        self.current_video_path = video_path
        self.current_ball_csv_path = ball_csv_path if ball_csv_path.exists() else None
        action4_json_path = video_path.with_name(f"{video_path.stem}_action4.json")
        self.current_action4_json_path = action4_json_path if action4_json_path.exists() else None
        self.current_ball_lookup = {}

        if project_json.exists():
            self._load_project_state(project_json)
            logger.info(f"Loaded existing project: {project_json}")
            return

        self.current_ball_data = self._read_ball_csv(self.current_ball_csv_path) if self.current_ball_csv_path else []
        self._rebuild_ball_lookup()
        if self.current_action4_json_path:
            self._load_initial_action4_json(self.current_action4_json_path)
        self._save_project_state()
        logger.info(f"Created project: {project_json}")

    def _load_project_state(self, project_json: Path):
        """Load project metadata and annotations from JSON."""
        with open(project_json, "r", encoding="utf-8") as file_obj:
            project_data = json.load(file_obj)

        self.current_ball_data = self._normalize_ball_data(project_data.get("ball_data", []))
        raw_ball_csv_path = project_data.get("ball_csv_path")
        if raw_ball_csv_path:
            candidate = Path(raw_ball_csv_path)
            self.current_ball_csv_path = candidate if candidate.exists() else self.current_ball_csv_path
        raw_action4_json_path = project_data.get("action4_json_path") or project_data.get("description", {}).get("action4_json_path")
        if raw_action4_json_path:
            candidate = Path(raw_action4_json_path)
            self.current_action4_json_path = candidate if candidate.exists() else self.current_action4_json_path

        annotations = project_data.get("annotations", {})
        self.annotations.load_state(annotations)
        self._rebuild_ball_lookup()

    def _save_project_state(self):
        """Persist current project state to JSON."""
        if not self.current_project_json or not self.current_video_path:
            return

        if self.current_ball_csv_path and not self.current_ball_data and self.current_ball_csv_path.exists():
            self.current_ball_data = self._read_ball_csv(self.current_ball_csv_path)
            self._rebuild_ball_lookup()

        payload = {
            "name": self.current_video_path.stem,
            "description": {
                "project_path": str(self.current_project_json),
                "video_path": str(self.current_video_path),
                "ball_csv_path": str(self.current_ball_csv_path) if self.current_ball_csv_path else None,
                "action4_json_path": str(self.current_action4_json_path) if self.current_action4_json_path else None,
            },
            "video_path": str(self.current_video_path),
            "ball_csv_path": str(self.current_ball_csv_path) if self.current_ball_csv_path else None,
            "action4_json_path": str(self.current_action4_json_path) if self.current_action4_json_path else None,
            "ball_data": self._serialize_ball_data(),
            "annotations": self.annotations.export_state(),
            "updated_at": self._now_iso(),
        }

        if self.current_project_json.exists():
            with open(self.current_project_json, "r", encoding="utf-8") as file_obj:
                existing_data = json.load(file_obj)
            payload["created_at"] = existing_data.get("created_at", payload["updated_at"])
        else:
            payload["created_at"] = payload["updated_at"]

        with open(self.current_project_json, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    @staticmethod
    def _read_ball_csv(csv_path: Path) -> list:
        """Read ball CSV rows as initial project data."""
        rows = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                try:
                    rows.append([
                        int(row.get("Frame", -1)),
                        int(row.get("Visibility", 0)),
                        int(float(row.get("X", -1))),
                        int(float(row.get("Y", -1))),
                    ])
                except (TypeError, ValueError):
                    continue
        return rows

    def _load_initial_action4_json(self, action4_json_path: Path):
        """Load neighboring *_action4.json as initial YOLO annotation state."""
        with open(action4_json_path, "r", encoding="utf-8") as file_obj:
            action_data = json.load(file_obj)

        raw_boxes = action_data.get("yolo_boxes", {})
        if raw_boxes:
            self.annotations.load_yolo_boxes(raw_boxes)

    def _rebuild_ball_lookup(self):
        """Reindex ball data rows by frame."""
        lookup = {}
        for row in self.current_ball_data:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                continue
            try:
                frame_idx = int(row[0])
            except (TypeError, ValueError):
                continue
            lookup[frame_idx] = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        self.current_ball_lookup = lookup

    def _get_current_ball_position_normalized(self):
        """Get current frame ball position normalized to source frame."""
        if not self.processor:
            return None

        row = self.current_ball_lookup.get(self.current_frame_idx)
        if not row:
            return None

        try:
            visibility = int(row[1])
            x = float(row[2])
            y = float(row[3])
        except (TypeError, ValueError, IndexError):
            return None

        if visibility <= 0 or x < 0 or y < 0 or self.processor.width <= 0 or self.processor.height <= 0:
            return None

        return (
            max(0.0, min(1.0, x / self.processor.width)),
            max(0.0, min(1.0, y / self.processor.height)),
        )

    def _upsert_ball_point(self, frame_idx: int, x: int, y: int):
        """Insert or update ball coordinates for a frame."""
        payload = [int(frame_idx), 1, int(x), int(y)]
        row = self.current_ball_lookup.get(frame_idx)
        if row is None:
            self.current_ball_data.append(payload)
            self.current_ball_lookup[frame_idx] = payload
        else:
            row[0] = int(frame_idx)
            row[1] = 1
            row[2] = int(x)
            row[3] = int(y)

    def _count_boxes_in_range(self, start_frame: int, end_frame: int) -> dict:
        """Return box totals for an inclusive frame range."""
        frames = 0
        boxes = 0
        for frame_idx, frame_boxes in self.annotations.yolo_boxes.items():
            if start_frame <= int(frame_idx) <= end_frame and frame_boxes:
                frames += 1
                boxes += len(frame_boxes)
        return {"frames": frames, "boxes": boxes}

    def _count_ball_points_in_range(self, start_frame: int, end_frame: int) -> int:
        """Return number of ball points inside an inclusive frame range."""
        return sum(
            1 for frame_idx in self.current_ball_lookup
            if start_frame <= int(frame_idx) <= end_frame
        )

    def _clear_ball_points_in_range(self, start_frame: int, end_frame: int) -> int:
        """Remove ball points inside an inclusive frame range."""
        remaining_ball_data = []
        removed = 0

        for row in self.current_ball_data:
            frame_idx = int(row[0])
            if start_frame <= frame_idx <= end_frame:
                removed += 1
                continue
            remaining_ball_data.append(row)

        if removed:
            self.current_ball_data = remaining_ball_data
            self._rebuild_ball_lookup()

        return removed

    def _focused_rally(self):
        """Return rally currently focused in the UI."""
        rally = None
        if self.selected_rally_id is not None:
            if self.selected_rally_id == 0:
                rally = self._pending_rally_preview()
            else:
                rally = next((item for item in self.annotations.rallies if item.get("id") == self.selected_rally_id), None)

        if rally is None:
            rally = next(
                (
                    item for item in self.annotations.rallies
                    if int(item.get("start_frame", -1)) <= self.current_frame_idx <= int(item.get("end_frame", -1))
                ),
                None,
            )

        if rally is None:
            rally = self._pending_rally_preview()

        return rally

    def _update_detail_timeline(self):
        """Show zoomed timeline for selected or current rally."""
        rally = self._focused_rally()
        self.detail_timeline.set_focus_rally(rally)
        if self.processor:
            self.detail_timeline.set_total_frames(self.processor.total_frames)
        self.detail_timeline.set_current_frame(self.current_frame_idx)

    def _serialize_ball_data(self) -> list:
        """Return ball data in compact array format."""
        normalized = self._normalize_ball_data(self.current_ball_data)
        return sorted(normalized, key=lambda row: row[0])

    @staticmethod
    def _normalize_ball_data(ball_data) -> list:
        """Normalize legacy and current ball data into [frame, visibility, x, y]."""
        normalized = []
        if not isinstance(ball_data, list):
            return normalized

        for row in ball_data:
            if isinstance(row, dict):
                try:
                    normalized.append([
                        int(row.get("Frame", -1)),
                        int(row.get("Visibility", 0)),
                        int(float(row.get("X", -1))),
                        int(float(row.get("Y", -1))),
                    ])
                except (TypeError, ValueError):
                    continue
            elif isinstance(row, (list, tuple)) and len(row) >= 4:
                try:
                    normalized.append([
                        int(row[0]),
                        int(row[1]),
                        int(row[2]),
                        int(row[3]),
                    ])
                except (TypeError, ValueError):
                    continue

        return normalized

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"
