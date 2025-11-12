"""Image canvas widget for displaying frames and editing bounding boxes."""
from typing import List, Tuple, Optional
from PyQt6.QtWidgets import QLabel, QMenu
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QMouseEvent
from loguru import logger


class ImageCanvas(QLabel):
    """Widget for displaying video frames with bounding box annotation support."""
    
    # Signals
    box_added = pyqtSignal(tuple)  # Emits (class_id, x_center, y_center, width, height)
    box_removed = pyqtSignal(int)  # Emits box index
    box_selected = pyqtSignal(int)  # Emits box index
    box_class_changed = pyqtSignal(int, int)  # Emits (box_index, new_class_id)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        
        # State
        self.current_pixmap: Optional[QPixmap] = None
        self.boxes: List[Tuple[int, float, float, float, float]] = []  # (cls, x_c, y_c, w, h) normalized
        self.selected_box_idx: Optional[int] = None
        self.box_colors = {0: QColor("#FF0000"), 1: QColor("#0000FF")}  # Default colors
        self.box_classes = {0: "Class 0", 1: "Class 1"}  # Default class names
        self.selected_color = QColor("#00FF00")  # Green
        
        # Drawing state
        self.is_drawing = False
        self.draw_start: Optional[QPoint] = None
        self.draw_end: Optional[QPoint] = None
        self.current_class_id = 0  # Default to Ball (class 0)
        
        # Editing state
        self.is_dragging = False
        self.drag_start: Optional[QPoint] = None
        self.drag_box_original: Optional[Tuple] = None  # Original box before drag
        self.resize_mode: Optional[str] = None  # 'tl', 'tr', 'bl', 'br', 'move'
        
        # Display settings
        self.show_boxes = True
        self.box_line_width = 2
        self.selected_line_width = 4
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_image(self, pixmap: QPixmap, boxes: Optional[List[Tuple]] = None):
        """
        Set the image to display with optional bounding boxes.
        
        Args:
            pixmap: QPixmap to display
            boxes: List of bounding boxes (class_id, x_center, y_center, width, height) normalized [0-1]
        """
        self.current_pixmap = pixmap
        self.boxes = boxes if boxes is not None else []
        self.selected_box_idx = None
        self.update()
    
    def set_box_colors(self, color_dict: dict):
        """Set colors for each class."""
        self.box_colors = {k: QColor(v) for k, v in color_dict.items()}
    
    def set_box_classes(self, class_dict: dict):
        """Set class names for each class ID."""
        self.box_classes = class_dict
    
    def set_current_class(self, class_id: int):
        """Set the current class for new boxes."""
        self.current_class_id = class_id
    
    def toggle_boxes_visibility(self, visible: bool):
        """Toggle bounding box visibility."""
        self.show_boxes = visible
        self.update()
    
    def paintEvent(self, event):
        """Override paint event to draw image and bounding boxes."""
        super().paintEvent(event)
        
        if self.current_pixmap is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate scaling to fit pixmap in widget
        widget_rect = self.rect()
        pixmap_size = self.current_pixmap.size()
        scaled_pixmap = self.current_pixmap.scaled(
            widget_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate offset to center the pixmap
        x_offset = (widget_rect.width() - scaled_pixmap.width()) // 2
        y_offset = (widget_rect.height() - scaled_pixmap.height()) // 2
        
        # Draw pixmap
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # Draw bounding boxes if enabled
        if self.show_boxes and self.boxes:
            self._draw_boxes(painter, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Draw current drawing box
        if self.is_drawing and self.draw_start and self.draw_end:
            pen = QPen(QColor("#FFFF00"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self.draw_start, self.draw_end).normalized()
            painter.drawRect(rect)
    
    def _draw_boxes(self, painter: QPainter, x_offset: int, y_offset: int, img_width: int, img_height: int):
        """Draw all bounding boxes."""
        for idx, (cls_id, x_c, y_c, w, h) in enumerate(self.boxes):
            # Convert normalized coordinates to pixel coordinates
            box_x = int((x_c - w/2) * img_width) + x_offset
            box_y = int((y_c - h/2) * img_height) + y_offset
            box_w = int(w * img_width)
            box_h = int(h * img_height)
            
            # Select color and line width
            if idx == self.selected_box_idx:
                color = self.selected_color
                line_width = self.selected_line_width
            else:
                color = self.box_colors.get(cls_id, QColor("#FFFFFF"))
                line_width = self.box_line_width
            
            # Draw rectangle
            pen = QPen(color, line_width)
            painter.setPen(pen)
            painter.drawRect(box_x, box_y, box_w, box_h)
            
            # Draw class label above box
            class_name = self.box_classes.get(cls_id, f"Class {cls_id}")
            painter.setFont(painter.font())
            painter.drawText(box_x, box_y - 5, class_name)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for box creation/selection."""
        pos = event.pos()
        
        # Check if clicking on existing box
        clicked_box_idx = self._find_box_at_position(pos)
        
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click: show context menu if clicking on box
            if clicked_box_idx is not None:
                self.selected_box_idx = clicked_box_idx
                self._show_box_context_menu(pos, clicked_box_idx)
            return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if clicked_box_idx is not None:
            # Select box
            self.selected_box_idx = clicked_box_idx
            self.box_selected.emit(clicked_box_idx)
            self.is_dragging = True
            self.drag_start = pos
            self.update()
        else:
            # Start drawing new box
            self.is_drawing = True
            self.draw_start = pos
            self.draw_end = pos
            self.selected_box_idx = None
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for box drawing/editing."""
        pos = event.pos()
        
        if self.is_drawing:
            self.draw_end = pos
            self.update()
        elif self.is_dragging and self.drag_start and self.selected_box_idx is not None:
            # Move the selected box
            if 0 <= self.selected_box_idx < len(self.boxes):
                delta_pos = pos - self.drag_start
                self._move_box(self.selected_box_idx, delta_pos)
                self.drag_start = pos
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to complete box creation."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if self.is_drawing and self.draw_start and self.draw_end:
            # Convert pixel coordinates to normalized coordinates
            rect = QRect(self.draw_start, self.draw_end).normalized()
            box = self._pixel_rect_to_normalized_box(rect)
            
            if box:  # Only add if valid
                self.boxes.append(box)
                self.box_added.emit(box)
            
            self.is_drawing = False
            self.draw_start = None
            self.draw_end = None
            self.update()
        
        self.is_dragging = False
        self.drag_start = None
    
    def keyPressEvent(self, event):
        """Handle key press for box deletion."""
        if event.key() == Qt.Key.Key_Delete and self.selected_box_idx is not None:
            if 0 <= self.selected_box_idx < len(self.boxes):
                removed_box = self.boxes.pop(self.selected_box_idx)
                self.box_removed.emit(self.selected_box_idx)
                self.selected_box_idx = None
                self.update()
                logger.debug(f"Box deleted: {removed_box}")
    
    def _find_box_at_position(self, pos: QPoint) -> Optional[int]:
        """Find box index at given position."""
        if not self.current_pixmap:
            return None
        
        # Get image display area
        widget_rect = self.rect()
        pixmap_size = self.current_pixmap.size()
        scaled_size = self.current_pixmap.scaled(
            widget_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).size()
        
        x_offset = (widget_rect.width() - scaled_size.width()) // 2
        y_offset = (widget_rect.height() - scaled_size.height()) // 2
        img_width = scaled_size.width()
        img_height = scaled_size.height()
        
        # Check each box
        for idx, (cls_id, x_c, y_c, w, h) in enumerate(self.boxes):
            box_x = int((x_c - w/2) * img_width) + x_offset
            box_y = int((y_c - h/2) * img_height) + y_offset
            box_w = int(w * img_width)
            box_h = int(h * img_height)
            
            if (box_x <= pos.x() <= box_x + box_w and
                box_y <= pos.y() <= box_y + box_h):
                return idx
        
        return None
    
    def _pixel_rect_to_normalized_box(self, rect: QRect) -> Optional[Tuple]:
        """Convert pixel rectangle to normalized YOLO box format."""
        if not self.current_pixmap or rect.width() < 5 or rect.height() < 5:
            return None
        
        # Get image display area
        widget_rect = self.rect()
        scaled_size = self.current_pixmap.scaled(
            widget_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).size()
        
        x_offset = (widget_rect.width() - scaled_size.width()) // 2
        y_offset = (widget_rect.height() - scaled_size.height()) // 2
        img_width = scaled_size.width()
        img_height = scaled_size.height()
        
        # Convert to image-relative coordinates
        x1 = max(0, rect.left() - x_offset)
        y1 = max(0, rect.top() - y_offset)
        x2 = min(img_width, rect.right() - x_offset)
        y2 = min(img_height, rect.bottom() - y_offset)
        
        # Normalize to [0, 1]
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Clamp to valid range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return (self.current_class_id, x_center, y_center, width, height)
    
    def _move_box(self, box_idx: int, delta: QPoint):
        """Move a bounding box by delta pixels."""
        if not self.current_pixmap or box_idx >= len(self.boxes):
            return
        
        # Get current box
        cls_id, x_c, y_c, w, h = self.boxes[box_idx]
        
        # Get image display dimensions
        widget_rect = self.rect()
        scaled_size = self.current_pixmap.scaled(
            widget_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).size()
        
        x_offset = (widget_rect.width() - scaled_size.width()) // 2
        y_offset = (widget_rect.height() - scaled_size.height()) // 2
        img_width = scaled_size.width()
        img_height = scaled_size.height()
        
        # Convert delta to normalized coordinates
        delta_x_norm = delta.x() / img_width
        delta_y_norm = delta.y() / img_height
        
        # Update box position
        new_x_c = x_c + delta_x_norm
        new_y_c = y_c + delta_y_norm
        
        # Clamp to valid range
        new_x_c = max(w/2, min(1.0 - w/2, new_x_c))
        new_y_c = max(h/2, min(1.0 - h/2, new_y_c))
        
        # Update box
        self.boxes[box_idx] = (cls_id, new_x_c, new_y_c, w, h)
        logger.debug(f"Box {box_idx} moved to ({new_x_c:.3f}, {new_y_c:.3f})")
    
    def _show_box_context_menu(self, pos: QPoint, box_idx: int):
        """Show context menu for bounding box operations."""
        if box_idx >= len(self.boxes):
            return
        
        menu = QMenu(self)
        
        # Change class submenu
        change_class_menu = menu.addMenu("Change Class")
        for class_id, class_name in sorted(self.box_classes.items()):
            action = change_class_menu.addAction(class_name)
            action.triggered.connect(lambda checked=False, cid=class_id: self._change_box_class(box_idx, cid))
        
        menu.addSeparator()
        
        # Delete action
        delete_action = menu.addAction("Delete Box")
        delete_action.triggered.connect(lambda: self._delete_box(box_idx))
        
        # Show menu at cursor position
        menu.exec(self.mapToGlobal(pos))
    
    def _change_box_class(self, box_idx: int, new_class_id: int):
        """Change the class of a bounding box."""
        if 0 <= box_idx < len(self.boxes):
            cls_id, x_c, y_c, w, h = self.boxes[box_idx]
            old_class = self.box_classes.get(cls_id, f"Class {cls_id}")
            new_class = self.box_classes.get(new_class_id, f"Class {new_class_id}")
            
            self.boxes[box_idx] = (new_class_id, x_c, y_c, w, h)
            self.box_class_changed.emit(box_idx, new_class_id)
            self.update()
            
            logger.info(f"Box {box_idx} class changed from '{old_class}' to '{new_class}'")
    
    def _delete_box(self, box_idx: int):
        """Delete a bounding box from context menu."""
        if 0 <= box_idx < len(self.boxes):
            removed_box = self.boxes.pop(box_idx)
            self.box_removed.emit(box_idx)
            self.selected_box_idx = None
            self.update()
            
            class_name = self.box_classes.get(removed_box[0], f"Class {removed_box[0]}")
            logger.info(f"Box deleted via context menu: {class_name}")
