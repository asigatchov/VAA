"""Main window for VAA application."""
from typing import Optional
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QCheckBox, QMenuBar, QMenu, QStatusBar
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
        self.annotations = AnnotationManager()
        self.yolo_tracker: Optional[YOLOTracker] = None
        
        # State
        self.current_frame_idx = 0
        self.is_playing = False
        self.show_superframe = self.ui_config.show_superframe_on_start
        self.show_boxes = self.ui_config.show_boxes_on_start
        self.playback_speed = self.ui_config.default_playback_speed
        
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
        left_panel.addWidget(self.canvas, stretch=1)
        
        # Control bar
        control_bar = QHBoxLayout()
        
        self.load_btn = QPushButton("📁 Load Video")
        self.load_btn.clicked.connect(self.load_video)
        control_bar.addWidget(self.load_btn)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_bar.addWidget(self.play_btn)
        
        self.prev_btn = QPushButton("|◀ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        control_bar.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("▶| Next")
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        control_bar.addWidget(self.next_btn)
        
        self.superframe_cb = QCheckBox("Show Superframe")
        self.superframe_cb.setChecked(self.show_superframe)
        self.superframe_cb.stateChanged.connect(self.toggle_superframe)
        control_bar.addWidget(self.superframe_cb)
        
        self.boxes_cb = QCheckBox("Show Boxes")
        self.boxes_cb.setChecked(self.show_boxes)
        self.boxes_cb.stateChanged.connect(self.toggle_boxes)
        control_bar.addWidget(self.boxes_cb)
        
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
        
        control_bar.addStretch()
        left_panel.addLayout(control_bar)
        
        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.set_action_colors(self.ui_config.action_colors)
        self.timeline.position_changed.connect(self.seek)
        left_panel.addWidget(self.timeline)
        
        main_layout.addLayout(left_panel, stretch=7)
        
        # Right panel - Action controls
        self.action_panel = ActionPanel(list(self.annot_config.action_types))
        self.action_panel.start_action_requested.connect(self.start_action)
        self.action_panel.end_action_requested.connect(self.end_action)
        self.action_panel.action_deleted.connect(self.delete_action)
        self.action_panel.seek_to_action.connect(self.seek)
        
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
        
        save_action = QAction("&Save Annotations", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.export_annotations)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        toggle_superframe_action = QAction("Toggle &Superframe", self)
        toggle_superframe_action.setShortcut("F1")
        toggle_superframe_action.triggered.connect(lambda: self.superframe_cb.toggle())
        view_menu.addAction(toggle_superframe_action)
        
        toggle_boxes_action = QAction("Toggle &Bounding Boxes", self)
        toggle_boxes_action.setShortcut("F2")
        toggle_boxes_action.triggered.connect(lambda: self.boxes_cb.toggle())
        view_menu.addAction(toggle_boxes_action)
        
        # Annotation menu
        annot_menu = menubar.addMenu("&Annotation")
        
        start_action_act = QAction("&Start Action", self)
        start_action_act.setShortcut("Return")
        start_action_act.triggered.connect(self.action_panel.start_btn.click)
        annot_menu.addAction(start_action_act)
        
        end_action_act = QAction("&End Action", self)
        end_action_act.setShortcut("Shift+Return")
        end_action_act.triggered.connect(self.action_panel.end_btn.click)
        annot_menu.addAction(end_action_act)
        
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
        prev_shortcut.triggered.connect(self.prev_frame)
        self.addAction(prev_shortcut)
        
        next_shortcut = QAction(self)
        next_shortcut.setShortcut("Right")
        next_shortcut.triggered.connect(self.next_frame)
        self.addAction(next_shortcut)
        
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
            # Release existing processor
            if self.processor:
                self.processor.release()
            
            # Load new video
            self.processor = VideoProcessor(
                file_path,
                target_size=self.annot_config.target_resolution,
                cache_limit=self.annot_config.cache_limit_frames
            )
            
            # Reset state
            self.current_frame_idx = 0
            self.is_playing = False
            self.annotations = AnnotationManager()
            
            # Update UI
            self.timeline.set_total_frames(self.processor.total_frames)
            self.timeline.set_current_frame(0)
            self.play_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Display first frame
            self.update_display()
            
            self.status_bar.showMessage(f"Loaded: {file_path} | {self.processor.total_frames} frames @ {self.processor.fps:.2f} fps")
            logger.info(f"Video loaded successfully: {file_path}")
            
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
            self.play_btn.setText("⏸ Pause")
            self.status_bar.showMessage("Playing...")
        else:
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.status_bar.showMessage("Paused")
    
    def prev_frame(self):
        """Go to previous frame."""
        if self.processor and self.current_frame_idx > 0:
            self.seek(self.current_frame_idx - 1)
    
    def next_frame(self):
        """Go to next frame."""
        if self.processor and self.current_frame_idx < self.processor.total_frames - 1:
            self.seek(self.current_frame_idx + 1)
        elif self.is_playing:
            # Stop playback at end
            self.toggle_play()
    
    def seek(self, frame_idx: int):
        """Seek to specific frame."""
        if not self.processor:
            return
        
        self.current_frame_idx = max(0, min(frame_idx, self.processor.total_frames - 1))
        self.timeline.set_current_frame(self.current_frame_idx)
        self.update_display()
    
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
    
    def on_class_selection_changed(self, index):
        """Handle class selection change for drawing new boxes."""
        class_id = self.class_combo.itemData(index)
        if class_id is not None:
            self.canvas.set_current_class(class_id)
            class_name = self.annot_config.box_classes.get(class_id, f"Class {class_id}")
            self.status_bar.showMessage(f"Drawing class set to: {class_name}")
    
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
        boxes = self.annotations.yolo_boxes.get(self.current_frame_idx, [])
        
        # Update canvas
        self.canvas.set_image(pixmap, boxes)
        
        # Update info labels
        self.update_info_labels()
    
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
    
    def start_action(self, action_type: str):
        """Start a new action annotation."""
        if not self.processor:
            return
        
        success = self.annotations.start_action(
            self.current_frame_idx,
            action_type,
            self.processor.fps
        )
        
        if success:
            self.status_bar.showMessage(f"Action started: {action_type}")
        else:
            QMessageBox.warning(
                self,
                "Action Already Started",
                "Please end the current action before starting a new one."
            )
    
    def end_action(self):
        """End the current action annotation."""
        if not self.processor:
            return
        
        action = self.annotations.end_action(self.current_frame_idx)
        
        if action:
            self.action_panel.set_actions(self.annotations.actions)
            self.timeline.set_actions(self.annotations.actions)
            self.status_bar.showMessage(f"Action completed: {action['type']}")
        else:
            QMessageBox.warning(
                self,
                "No Action to End",
                "Please start an action first."
            )
    
    def delete_action(self, action_id: int):
        """Delete an action."""
        reply = QMessageBox.question(
            self,
            "Delete Action",
            "Are you sure you want to delete this action?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.annotations.delete_action(action_id):
                self.action_panel.set_actions(self.annotations.actions)
                self.timeline.set_actions(self.annotations.actions)
                self.status_bar.showMessage("Action deleted")
    
    def on_box_added(self, box):
        """Handle box added event from canvas."""
        self.annotations.add_yolo_box(self.current_frame_idx, box)
        self.status_bar.showMessage(f"Box added to frame {self.current_frame_idx}")
        logger.debug(f"Box added: {box}")
    
    def on_box_removed(self, box_index):
        """Handle box removed event from canvas."""
        self.annotations.remove_yolo_box(self.current_frame_idx, box_index)
        self.status_bar.showMessage(f"Box removed from frame {self.current_frame_idx}")
    
    def on_box_class_changed(self, box_index: int, new_class_id: int):
        """Handle box class change event from canvas."""
        # Update the box in annotations
        if self.current_frame_idx in self.annotations.yolo_boxes:
            boxes = self.annotations.yolo_boxes[self.current_frame_idx]
            if 0 <= box_index < len(boxes):
                old_box = boxes[box_index]
                # Update class while keeping coordinates
                new_box = (new_class_id, old_box[1], old_box[2], old_box[3], old_box[4])
                boxes[box_index] = new_box
                
                class_name = self.annot_config.box_classes.get(new_class_id, f"Class {new_class_id}")
                self.status_bar.showMessage(f"Box class changed to {class_name}")
                logger.info(f"Frame {self.current_frame_idx}, Box {box_index} class changed to {class_name}")
    
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
        
        count = self.canvas.paste_boxes()
        if count > 0:
            # Add pasted boxes to annotations
            for box in self.canvas.boxes[-count:]:  # Get last 'count' boxes
                self.annotations.add_yolo_box(self.current_frame_idx, box)
            
            self.status_bar.showMessage(f"Pasted {count} boxes to frame {self.current_frame_idx}")
            self.update_display()
        else:
            self.status_bar.showMessage("No boxes in clipboard to paste")
    
    # Export methods
    
    def export_annotations(self):
        """Export annotations to files."""
        if not self.processor:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        # Check if there are annotations
        stats = self.annotations.get_statistics()
        if stats["total_actions"] == 0 and stats["total_boxes"] == 0:
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
            ""
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
                f"Actions: {action_count} items\n\n"
                f"Location: {output_dir}"
            )
            
            self.status_bar.showMessage(f"Export complete: {output_dir}")
            logger.info(f"Export successful: {label_count} labels, {action_count} actions")
            
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
        if self.annotations.actions or self.annotations.yolo_boxes:
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
