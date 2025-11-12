"""Timeline widget with action markers."""
from typing import List, Dict
from PyQt6.QtWidgets import QWidget, QSlider
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen


class TimelineWidget(QWidget):
    """Custom timeline widget with action range visualization."""
    
    # Signals
    position_changed = pyqtSignal(int)  # Emits new frame position
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        
        # State
        self.total_frames = 0
        self.current_frame = 0
        self.actions: List[Dict] = []
        self.action_colors = {}
        
        # Interaction
        self.is_dragging = False
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_total_frames(self, total_frames: int):
        """Set the total number of frames."""
        self.total_frames = max(1, total_frames)
        self.update()
    
    def set_current_frame(self, frame: int):
        """Set the current frame position."""
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        self.update()
    
    def set_actions(self, actions: List[Dict]):
        """Set the list of actions to visualize."""
        self.actions = actions
        self.update()
    
    def set_action_colors(self, colors: Dict[str, str]):
        """Set colors for action types."""
        self.action_colors = {k: QColor(v) for k, v in colors.items()}
    
    def paintEvent(self, event):
        """Paint the timeline with action markers."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Draw background
        painter.fillRect(rect, QColor("#3c3c3c"))
        
        # Draw action ranges
        if self.total_frames > 0:
            action_y = 10
            action_height = 20
            
            for action in self.actions:
                start_frame = action.get("start_frame", 0)
                end_frame = action.get("end_frame", 0)
                action_type = action.get("type", "")
                
                # Calculate pixel positions
                start_x = int((start_frame / self.total_frames) * width)
                end_x = int((end_frame / self.total_frames) * width)
                
                # Draw action bar
                color = self.action_colors.get(action_type, QColor("#888888"))
                painter.fillRect(start_x, action_y, end_x - start_x, action_height, color)
        
            # Draw timeline bar
            timeline_y = 40
            timeline_height = 10
            painter.fillRect(0, timeline_y, width, timeline_height, QColor("#555555"))
            
            # Draw current position marker
            if self.total_frames > 0:
                pos_x = int((self.current_frame / self.total_frames) * width)
                painter.setPen(QPen(QColor("#00FF00"), 3))
                painter.drawLine(pos_x, 0, pos_x, height)
                
                # Draw position diamond
                points = [
                    (pos_x, timeline_y - 5),
                    (pos_x + 5, timeline_y),
                    (pos_x, timeline_y + 5),
                    (pos_x - 5, timeline_y)
                ]
                painter.setBrush(QColor("#00FF00"))
                painter.drawPolygon([QPoint(x, y) for x, y in points])
    
    def mousePressEvent(self, event):
        """Handle mouse press to seek."""
        if event.button() == Qt.MouseButton.LeftButton and self.total_frames > 0:
            self.is_dragging = True
            self._seek_to_position(event.pos().x())
    
    def mouseMoveEvent(self, event):
        """Handle mouse move during drag."""
        if self.is_dragging and self.total_frames > 0:
            self._seek_to_position(event.pos().x())
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
    
    def _seek_to_position(self, x_pos: int):
        """Seek to frame based on x position."""
        width = self.width()
        if width > 0:
            frame = int((x_pos / width) * self.total_frames)
            frame = max(0, min(frame, self.total_frames - 1))
            if frame != self.current_frame:
                self.current_frame = frame
                self.position_changed.emit(frame)
                self.update()
