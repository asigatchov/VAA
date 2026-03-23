"""Timeline widget with action markers."""
from typing import List, Dict, Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen


class TimelineWidget(QWidget):
    """Custom timeline widget with action range visualization."""
    
    # Signals
    position_changed = pyqtSignal(int)  # Emits new frame position
    
    def __init__(self, parent=None, detail_mode: bool = False):
        super().__init__(parent)
        self.detail_mode = detail_mode
        if self.detail_mode:
            self.setMinimumHeight(90)
            self.setMaximumHeight(120)
        else:
            self.setMinimumHeight(60)
            self.setMaximumHeight(80)
        
        # State
        self.total_frames = 0
        self.current_frame = 0
        self.actions: List[Dict] = []
        self.action_colors = {}
        self.focus_rally: Optional[Dict] = None
        
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

    def set_focus_rally(self, rally: Optional[Dict]):
        """Set rally used for detail mode."""
        self.focus_rally = rally
        self.update()
    
    def paintEvent(self, event):
        """Paint the timeline with action markers."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        # Draw background
        painter.fillRect(rect, QColor("#3c3c3c"))

        if self.detail_mode:
            self._paint_detail_timeline(painter, rect)
        else:
            self._paint_main_timeline(painter, rect)

    def _paint_main_timeline(self, painter: QPainter, rect):
        """Paint full timeline."""
        width = rect.width()
        height = rect.height()
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

    def _paint_detail_timeline(self, painter: QPainter, rect):
        """Paint zoomed timeline for current or selected rally."""
        width = rect.width()
        height = rect.height()

        if not self.focus_rally:
            painter.setPen(QColor("#bdbdbd"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No active rally")
            return

        start_frame = int(self.focus_rally.get("start_frame", 0))
        end_frame = int(self.focus_rally.get("end_frame", start_frame + 1))
        span = max(1, end_frame - start_frame)

        minor_step = max(1, span // 12)
        major_step = max(1, span // 4)

        painter.setPen(QPen(QColor(59, 130, 246, 80), 1))
        for frame in range(start_frame, end_frame + 1, minor_step):
            x = int(((frame - start_frame) / span) * width)
            painter.drawLine(x, 0, x, height)

        painter.setPen(QPen(QColor(59, 130, 246, 160), 1))
        for frame in range(start_frame, end_frame + 1, major_step):
            x = int(((frame - start_frame) / span) * width)
            painter.drawLine(x, 0, x, height)

        painter.fillRect(0, 20, width, 24, QColor("#B9F27C"))

        for action in self.focus_rally.get("actions", []):
            action_start = int(action.get("start_frame", start_frame))
            action_end = int(action.get("end_frame", action_start))
            left = int(((action_start - start_frame) / span) * width)
            right = int(((action_end - start_frame) / span) * width)
            bar_width = max(2, right - left)
            color = self.action_colors.get(action.get("type", ""), QColor("#60a5fa"))
            painter.fillRect(left, 54, bar_width, 18, color)
            painter.setPen(QColor("#f5f5f5"))
            painter.drawText(left + 2, 50, action.get("type", ""))

        pos_x = int(((self.current_frame - start_frame) / span) * width)
        pos_x = max(0, min(width, pos_x))
        painter.setPen(QPen(QColor("#ef4444"), 2))
        painter.drawLine(pos_x, 0, pos_x, height)

        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(6, 14, f"Rally {start_frame}-{end_frame}")
        painter.drawText(max(6, width - 100), 14, f"Frame {self.current_frame}")
    
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
            if self.detail_mode and self.focus_rally:
                start_frame = int(self.focus_rally.get("start_frame", 0))
                end_frame = int(self.focus_rally.get("end_frame", start_frame + 1))
                span = max(1, end_frame - start_frame)
                frame = start_frame + int((x_pos / width) * span)
                frame = max(start_frame, min(frame, end_frame))
            else:
                frame = int((x_pos / width) * self.total_frames)
                frame = max(0, min(frame, self.total_frames - 1))
            if frame != self.current_frame:
                self.current_frame = frame
                self.position_changed.emit(frame)
                self.update()
