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
    box_class_changed = pyqtSignal(int, int)  # Emits (box_id_or_index, new_class_id)
    box_geometry_changed = pyqtSignal(int, float, float, float, float, int)  # box_index, x, y, w, h, box_id
    ball_point_set = pyqtSignal(float, float)  # Emits normalized x, y
    assistant_click_requested = pyqtSignal(float, float)  # Emits normalized x, y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        
        # State
        self.current_pixmap: Optional[QPixmap] = None
        self.boxes: List[Tuple] = []  # (cls, x_c, y_c, w, h, box_id) normalized
        self.selected_box_idx: Optional[int] = None
        self.box_colors = {0: QColor("#FF0000"), 1: QColor("#0000FF")}  # Default colors
        self.box_classes = {0: "Class 0", 1: "Class 1"}  # Default class names
        self.selected_color = QColor("#00FF00")  # Green
        self.confirmed_box_ids: set[int] = set()
        
        # Drawing state
        self.is_drawing = False
        self.draw_start: Optional[QPoint] = None
        self.draw_end: Optional[QPoint] = None
        self.current_class_id = 0  # Default to Ball (class 0)
        self.last_used_class_id = 0  # Remember last used class for continuity
        
        # Editing state
        self.is_dragging = False
        self.drag_start: Optional[QPoint] = None
        self.drag_box_original: Optional[Tuple] = None  # Original box before drag
        self.resize_mode: Optional[str] = None  # 'tl', 'tr', 'bl', 'br', 'move', None
        self.is_resizing = False
        self.corner_grab_distance = 8  # Click target for resize handle
        self.handle_radius = 3
        self.min_box_size_px = 3
        
        # Clipboard for copy/paste
        self.clipboard_boxes: List[Tuple] = []
        
        # Display settings
        self.show_boxes = True
        self.visible_box_class_ids: Optional[set[int]] = None
        self.show_ball = True
        self.ball_markup_mode = False
        self.ball_position: Optional[Tuple[float, float]] = None
        self.box_line_width = 2
        self.selected_line_width = 4
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_image(self, pixmap: QPixmap, boxes: Optional[List[Tuple]] = None, ball_position: Optional[Tuple[float, float]] = None):
        """
        Set the image to display with optional bounding boxes.
        
        Args:
            pixmap: QPixmap to display
            boxes: List of bounding boxes (class_id, x_center, y_center, width, height, box_id) normalized [0-1]
        """
        self.current_pixmap = pixmap
        self.boxes = boxes if boxes is not None else []
        self.ball_position = ball_position
        self.selected_box_idx = None
        self.update()

    def set_confirmed_box_ids(self, box_ids: set[int]):
        """Mark boxes confirmed for assistant initialization."""
        self.confirmed_box_ids = set(box_ids)
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
        self.last_used_class_id = class_id  # Remember for next box
    
    def toggle_boxes_visibility(self, visible: bool):
        """Toggle bounding box visibility."""
        self.show_boxes = visible
        if not visible:
            self.selected_box_idx = None
        self.update()

    def set_visible_box_classes(self, class_ids: Optional[set[int]]):
        """Restrict visible/editable boxes to the provided class ids."""
        self.visible_box_class_ids = None if class_ids is None else set(class_ids)
        if self.selected_box_idx is not None and (
            self.selected_box_idx >= len(self.boxes) or
            not self._is_box_visible(self.boxes[self.selected_box_idx])
        ):
            self.selected_box_idx = None
        self.update()

    def toggle_ball_visibility(self, visible: bool):
        """Toggle ball marker visibility."""
        self.show_ball = visible
        self.update()

    def set_ball_markup_mode(self, enabled: bool):
        """Enable or disable ball markup mode."""
        self.ball_markup_mode = enabled

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

        if self.show_ball and self.ball_position is not None:
            self._draw_ball_marker(painter, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Draw current drawing box
        if self.is_drawing and self.draw_start and self.draw_end:
            pen = QPen(QColor("#FFFF00"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self.draw_start, self.draw_end).normalized()
            painter.drawRect(rect)
    
    def _draw_boxes(self, painter: QPainter, x_offset: int, y_offset: int, img_width: int, img_height: int):
        """Draw all bounding boxes."""
        for idx, box in enumerate(self.boxes):
            if not self._is_box_visible(box):
                continue
            # Unpack box (handle both old 5-element and new 6-element format)
            if len(box) == 6:
                cls_id, x_c, y_c, w, h, box_id = box
            else:
                cls_id, x_c, y_c, w, h = box
                box_id = None
                
            # Convert normalized coordinates to pixel coordinates
            box_x = int((x_c - w/2) * img_width) + x_offset
            box_y = int((y_c - h/2) * img_height) + y_offset
            box_w = int(w * img_width)
            box_h = int(h * img_height)
            
            # Select color and line width
            is_confirmed = box_id is not None and int(box_id) in self.confirmed_box_ids
            if idx == self.selected_box_idx:
                color = self.selected_color
                line_width = self.selected_line_width
                pen_style = Qt.PenStyle.SolidLine
            elif is_confirmed:
                color = self.selected_color
                line_width = self.selected_line_width
                pen_style = Qt.PenStyle.DashLine
            else:
                color = self.box_colors.get(cls_id, QColor("#FFFFFF"))
                line_width = self.box_line_width
                pen_style = Qt.PenStyle.SolidLine
            
            # Draw rectangle
            pen = QPen(color, line_width, pen_style)
            painter.setPen(pen)
            painter.drawRect(box_x, box_y, box_w, box_h)
            if idx == self.selected_box_idx:
                self._draw_resize_handles(painter, color, box_x, box_y, box_w, box_h)
            
            # Draw class label with ID above box
            class_name = self.box_classes.get(cls_id, f"Class {cls_id}")
            if box_id is not None:
                label_text = f"{class_name} #{box_id}"
            else:
                label_text = class_name
            painter.setFont(painter.font())
            painter.drawText(box_x, box_y - 5, label_text)

    def _draw_resize_handles(
        self,
        painter: QPainter,
        color: QColor,
        box_x: int,
        box_y: int,
        box_w: int,
        box_h: int,
    ):
        """Draw visible resize handles for easier corner grabbing."""
        painter.save()
        painter.setPen(QPen(color, 1))
        painter.setBrush(color)
        radius = self.handle_radius
        painter.drawEllipse(QPoint(box_x, box_y), radius, radius)
        painter.drawEllipse(QPoint(box_x + box_w, box_y + box_h), radius, radius)
        painter.restore()

    def _draw_ball_marker(self, painter: QPainter, x_offset: int, y_offset: int, img_width: int, img_height: int):
        """Draw the current ball position."""
        if self.ball_position is None:
            return

        x_norm, y_norm = self.ball_position
        ball_x = int(x_norm * img_width) + x_offset
        ball_y = int(y_norm * img_height) + y_offset
        painter.setPen(QPen(QColor("#FFD60A"), 2))
        painter.setBrush(QColor("#FFD60A"))
        painter.drawEllipse(QPoint(ball_x, ball_y), 5, 5)
        painter.drawText(ball_x + 8, ball_y - 8, "Ball")
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for box creation/selection/resizing."""
        pos = event.pos()
        modifiers = event.modifiers()

        if event.button() == Qt.MouseButton.LeftButton and bool(modifiers & Qt.KeyboardModifier.ShiftModifier):
            click_point = self._point_to_normalized(pos)
            if click_point is not None:
                self.assistant_click_requested.emit(click_point[0], click_point[1])
            return

        if event.button() == Qt.MouseButton.LeftButton and self.ball_markup_mode:
            ball_point = self._point_to_normalized(pos)
            if ball_point is not None:
                self.ball_position = ball_point
                self.ball_point_set.emit(ball_point[0], ball_point[1])
                self.update()
            return
        
        # Check if clicking on existing box
        clicked_box_idx = self._find_box_at_position(pos)
        
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click: show context menu if clicking on box
            if clicked_box_idx is not None:
                self.selected_box_idx = clicked_box_idx
                self._show_box_context_menu(pos, clicked_box_idx)
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            if clicked_box_idx is not None:
                self.selected_box_idx = clicked_box_idx
                self._delete_box(clicked_box_idx)
            return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if clicked_box_idx is not None:
            corner = self._find_box_corner_at_position(pos, clicked_box_idx)
            if corner:
                self.selected_box_idx = clicked_box_idx
                self.is_resizing = True
                self.resize_mode = corner
                self.drag_start = pos
                self.drag_box_original = self.boxes[clicked_box_idx]
                logger.debug(f"Resizing box {clicked_box_idx} via handle: {corner}")
            else:
                # Select and move box
                self.selected_box_idx = clicked_box_idx
                self.box_selected.emit(clicked_box_idx)
                self.is_dragging = True
                self.drag_start = pos
            self.update()
        else:
            # Start drawing new box with last used class
            self.is_drawing = True
            self.draw_start = pos
            self.draw_end = pos
            self.selected_box_idx = None
            # Use last used class for continuity
            self.current_class_id = self.last_used_class_id
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for box drawing/editing/resizing."""
        pos = event.pos()
        
        if self.is_drawing:
            self.draw_end = pos
            self.update()
        elif self.is_resizing and self.drag_start and self.selected_box_idx is not None:
            # Resize the selected box
            if 0 <= self.selected_box_idx < len(self.boxes):
                delta_pos = pos - self.drag_start
                self._resize_box(self.selected_box_idx, delta_pos)
                self.update()
        elif self.is_dragging and self.drag_start and self.selected_box_idx is not None:
            # Move the selected box
            if 0 <= self.selected_box_idx < len(self.boxes):
                delta_pos = pos - self.drag_start
                self._move_box(self.selected_box_idx, delta_pos)
                self.drag_start = pos
                self.update()
        else:
            hovered_box_idx = self._find_box_at_position(pos)
            if hovered_box_idx is not None and self._find_box_corner_at_position(pos, hovered_box_idx):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif hovered_box_idx is not None:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
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
                # Remember this class for next box
                self.last_used_class_id = box[0]
                self.current_class_id = box[0]
            
            self.is_drawing = False
            self.draw_start = None
            self.draw_end = None
            self.update()
        
        self.is_dragging = False
        self.is_resizing = False
        self.resize_mode = None
        self.drag_start = None
        self.drag_box_original = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.selected_box_idx is not None and 0 <= self.selected_box_idx < len(self.boxes):
            box = self.boxes[self.selected_box_idx]
            if len(box) == 6:
                cls_id, x_c, y_c, w, h, box_id = box
                self.box_geometry_changed.emit(self.selected_box_idx, x_c, y_c, w, h, box_id)
    
    def keyPressEvent(self, event):
        """Handle key press for box deletion."""
        if event.key() == Qt.Key.Key_Delete and self.selected_box_idx is not None:
            if 0 <= self.selected_box_idx < len(self.boxes):
                removed_box = self.boxes[self.selected_box_idx]
                # Emit the box_id if available for deletion by ID
                if len(removed_box) == 6:
                    box_id = removed_box[5]
                    self.box_removed.emit(box_id)  # Emit box_id instead of index
                    logger.debug(f"Box deleted by id: {box_id}")
                else:
                    self.box_removed.emit(self.selected_box_idx)  # Fallback to index
                    logger.debug(f"Box deleted by index: {self.selected_box_idx}")
                
                self.boxes.pop(self.selected_box_idx)
                self.selected_box_idx = None
                self.update()
    
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
        for idx, box in enumerate(self.boxes):
            if not self._is_box_visible(box):
                continue
            # Handle both 5 and 6 element boxes
            if len(box) == 6:
                cls_id, x_c, y_c, w, h, box_id = box
            else:
                cls_id, x_c, y_c, w, h = box
                
            box_x = int((x_c - w/2) * img_width) + x_offset
            box_y = int((y_c - h/2) * img_height) + y_offset
            box_w = int(w * img_width)
            box_h = int(h * img_height)
            
            if (box_x <= pos.x() <= box_x + box_w and
                box_y <= pos.y() <= box_y + box_h):
                return idx
        
        return None

    def _is_box_visible(self, box: Tuple) -> bool:
        """Return True if a box should be drawn and editable under current filters."""
        if self.visible_box_class_ids is None:
            return True
        cls_id = int(box[0])
        return cls_id in self.visible_box_class_ids
    
    def _pixel_rect_to_normalized_box(self, rect: QRect) -> Optional[Tuple]:
        """Convert pixel rectangle to normalized YOLO box format (without ID - will be assigned later)."""
        if not self.current_pixmap or rect.width() < self.min_box_size_px or rect.height() < self.min_box_size_px:
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
        
        # Return 5-element tuple (ID will be assigned by annotation manager)
        return (self.current_class_id, x_center, y_center, width, height)

    def _point_to_normalized(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        """Convert widget coordinates to normalized image coordinates."""
        if not self.current_pixmap:
            return None

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
        if img_width <= 0 or img_height <= 0:
            return None

        x = pos.x() - x_offset
        y = pos.y() - y_offset
        if x < 0 or y < 0 or x > img_width or y > img_height:
            return None

        return (
            max(0.0, min(1.0, x / img_width)),
            max(0.0, min(1.0, y / img_height)),
        )
    
    def _move_box(self, box_idx: int, delta: QPoint):
        """Move a bounding box by delta pixels."""
        if not self.current_pixmap or box_idx >= len(self.boxes):
            return
        
        # Get current box
        box = self.boxes[box_idx]
        if len(box) == 6:
            cls_id, x_c, y_c, w, h, box_id = box
        else:
            cls_id, x_c, y_c, w, h = box
            box_id = None
        
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
        
        # Update box (preserve ID if exists)
        if box_id is not None:
            self.boxes[box_idx] = (cls_id, new_x_c, new_y_c, w, h, box_id)
        else:
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
            box = self.boxes[box_idx]
            if len(box) == 6:
                cls_id, x_c, y_c, w, h, box_id = box
            else:
                cls_id, x_c, y_c, w, h = box
                box_id = None
                
            old_class = self.box_classes.get(cls_id, f"Class {cls_id}")
            new_class = self.box_classes.get(new_class_id, f"Class {new_class_id}")
            
            # Update box (preserve ID if exists)
            if box_id is not None:
                self.boxes[box_idx] = (new_class_id, x_c, y_c, w, h, box_id)
            else:
                self.boxes[box_idx] = (new_class_id, x_c, y_c, w, h)
            self.box_class_changed.emit(box_id if box_id is not None else box_idx, new_class_id)
            self.update()
            self.last_used_class_id = new_class_id       
            logger.info(f"Box {box_idx} class changed from '{old_class}' to '{new_class}'")
    
    def _delete_box(self, box_idx: int):
        """Delete a bounding box from context menu."""
        if 0 <= box_idx < len(self.boxes):
            removed_box = self.boxes.pop(box_idx)
            
            # Emit box_id if available, otherwise index
            if len(removed_box) == 6:
                box_id = removed_box[5]
                self.box_removed.emit(box_id)
                class_name = self.box_classes.get(removed_box[0], f"Class {removed_box[0]}")
                logger.info(f"Box deleted via context menu: {class_name} #{box_id}")
            else:
                self.box_removed.emit(box_idx)
                class_name = self.box_classes.get(removed_box[0], f"Class {removed_box[0]}")
                logger.info(f"Box deleted via context menu: {class_name}")
            
            self.selected_box_idx = None
            self.update()
    
    def _find_box_corner_at_position(self, pos: QPoint, box_idx: int) -> Optional[str]:
        """Find which corner of a box is near the position (for resizing).
        
        Returns:
            'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right), or None
        """
        if not self.current_pixmap or box_idx >= len(self.boxes):
            return None
        
        box = self.boxes[box_idx]
        if len(box) == 6:
            cls_id, x_c, y_c, w, h, box_id = box
        else:
            cls_id, x_c, y_c, w, h = box
        
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
        
        # Calculate box corners in pixel coordinates
        box_x1 = int((x_c - w/2) * img_width) + x_offset
        box_y1 = int((y_c - h/2) * img_height) + y_offset
        box_x2 = int((x_c + w/2) * img_width) + x_offset
        box_y2 = int((y_c + h/2) * img_height) + y_offset
        
        mouse_x = pos.x()
        mouse_y = pos.y()
        grab_dist = self.corner_grab_distance
        
        # Visible handles are top-left and bottom-right only.
        if abs(mouse_x - box_x1) <= grab_dist and abs(mouse_y - box_y1) <= grab_dist:
            return 'tl'  # Top-left
        elif abs(mouse_x - box_x2) <= grab_dist and abs(mouse_y - box_y2) <= grab_dist:
            return 'br'  # Bottom-right
        
        return None
    
    def _resize_box(self, box_idx: int, delta: QPoint):
        """Resize a bounding box by dragging a corner."""
        if not self.current_pixmap or box_idx >= len(self.boxes) or not self.drag_box_original:
            return
        
        if not self.resize_mode:
            return
        
        # Get original box
        if len(self.drag_box_original) == 6:
            cls_id, orig_x_c, orig_y_c, orig_w, orig_h, box_id = self.drag_box_original
        else:
            cls_id, orig_x_c, orig_y_c, orig_w, orig_h = self.drag_box_original
            box_id = None
        
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
        
        # Calculate new box dimensions based on which corner is being dragged
        new_x_c = orig_x_c
        new_y_c = orig_y_c
        new_w = orig_w
        new_h = orig_h
        
        if self.resize_mode == 'tl':  # Top-left handle
            new_w = orig_w - delta_x_norm
            new_h = orig_h - delta_y_norm
            new_x_c = orig_x_c + delta_x_norm / 2
            new_y_c = orig_y_c + delta_y_norm / 2
        elif self.resize_mode == 'br':  # Bottom-right handle
            new_w = orig_w + delta_x_norm
            new_h = orig_h + delta_y_norm
            new_x_c = orig_x_c + delta_x_norm / 2
            new_y_c = orig_y_c + delta_y_norm / 2
        
        # Use a pixel-based minimum so very small objects like the ball remain annotatable.
        min_w = self.min_box_size_px / max(1, img_width)
        min_h = self.min_box_size_px / max(1, img_height)
        new_w = max(min_w, min(1.0, new_w))
        new_h = max(min_h, min(1.0, new_h))
        
        # Clamp center position
        new_x_c = max(new_w/2, min(1.0 - new_w/2, new_x_c))
        new_y_c = max(new_h/2, min(1.0 - new_h/2, new_y_c))
        
        # Update box (preserve ID if exists)
        if box_id is not None:
            self.boxes[box_idx] = (cls_id, new_x_c, new_y_c, new_w, new_h, box_id)
        else:
            self.boxes[box_idx] = (cls_id, new_x_c, new_y_c, new_w, new_h)
        logger.debug(f"Box {box_idx} resized to ({new_w:.3f}, {new_h:.3f})")
    
    def copy_boxes(self):
        """Copy all boxes from current frame to clipboard."""
        self.clipboard_boxes = [box for box in self.boxes]
        logger.info(f"Copied {len(self.clipboard_boxes)} boxes to clipboard")
        return len(self.clipboard_boxes)
    
    def paste_boxes(self):
        """Paste boxes from clipboard to current frame."""
        if not self.clipboard_boxes:
            logger.warning("No boxes in clipboard to paste")
            return 0
        
        # Важно: не добавлять боксы напрямую в self.boxes!
        # Вместо этого просто вернуть количество для статуса
        pasted_count = len(self.clipboard_boxes)
        self.update()  # Только обновить отображение, но не менять состояние
        logger.info(f"Ready to paste {pasted_count} boxes (will be added via annotation manager)")
        return pasted_count

    # def paste_boxes(self):
    #     """Paste boxes from clipboard to current frame (without emitting signals).
        
    #     Note: This only updates the visual display. The main window is responsible
    #     for adding boxes to the annotation manager.
    #     """
    #     if not self.clipboard_boxes:
    #         logger.warning("No boxes in clipboard to paste")
    #         return 0
        
    #     # Add all clipboard boxes to current frame (visual only)
    #     pasted_count = 0
    #     for box in self.clipboard_boxes:
    #         self.boxes.append(box)
    #         pasted_count += 1
        
    #     self.update()
    #     logger.info(f"Pasted {pasted_count} boxes to canvas (visual only)")
    #     return pasted_count
