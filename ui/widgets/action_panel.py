"""Action panel widget for managing rally annotations."""
from typing import List, Dict

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QAbstractItemView, QComboBox, QFrame, QMenu, QMessageBox, QSpinBox, QTreeWidget, QTreeWidgetItem
)


class ActionPanel(QWidget):
    """Panel for rally management and step selection."""

    start_rally_requested = pyqtSignal()
    end_rally_requested = pyqtSignal()
    cancel_rally_requested = pyqtSignal()
    action_type_selected = pyqtSignal(str)
    action_reclass_requested = pyqtSignal(int, int, int, str, str)
    rally_deleted = pyqtSignal(int)
    rally_selected = pyqtSignal(int)
    seek_to_rally = pyqtSignal(int)
    playback_step_changed = pyqtSignal(int)

    def __init__(self, action_types: List[str], parent=None):
        super().__init__(parent)
        self.action_types = action_types
        self.rallies: List[Dict] = []
        self.pending_rally: Dict | None = None
        self.is_rally_pending = False
        self.selected_action_type = self.action_types[0] if self.action_types else ""
        self.action_buttons: Dict[str, QPushButton] = {}
        self.current_frame = 0
        self.focus_rally_id: int | None = None
        self.selected_action_item: Dict | None = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        selector_label = QLabel("Rally Flow")
        selector_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(selector_label)

        steps_frame = QFrame()
        steps_frame.setStyleSheet(
            "QFrame { background-color: #1f2329; border: 1px solid #31363f; border-radius: 10px; }"
        )
        steps_layout = QHBoxLayout(steps_frame)
        steps_layout.setContentsMargins(8, 8, 8, 8)
        steps_layout.setSpacing(6)
        steps_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for idx, action_type in enumerate(self.action_types):
            button = QPushButton(action_type)
            button.setCheckable(True)
            button.setMinimumHeight(28)
            button.setMinimumWidth(68)
            button.clicked.connect(lambda checked, value=action_type: self._select_action_type(value))
            steps_layout.addWidget(button)
            self.action_buttons[action_type] = button

            if idx < len(self.action_types) - 1:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #6c757d; font-size: 14px; font-weight: bold;")
                steps_layout.addWidget(arrow)

        layout.addWidget(steps_frame)

        self.selected_type_label = QLabel("")
        self.selected_type_label.setStyleSheet("color: #cfd8dc; font-size: 11px;")
        layout.addWidget(self.selected_type_label)
        self._update_action_buttons()

        playback_layout = QHBoxLayout()
        playback_layout.setContentsMargins(0, 2, 0, 2)
        playback_layout.setSpacing(8)
        playback_label = QLabel("Step")
        playback_label.setStyleSheet("font-weight: bold;")
        playback_layout.addWidget(playback_label)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 30)
        self.step_spin.setValue(1)
        self.step_spin.setMinimumWidth(84)
        self.step_spin.setMaximumWidth(96)
        self.step_spin.valueChanged.connect(self.playback_step_changed.emit)
        playback_layout.addWidget(self.step_spin)

        playback_hint = QLabel("frames")
        playback_hint.setStyleSheet("color: #8fa1b7; font-size: 11px;")
        playback_layout.addWidget(playback_hint)

        playback_layout.addStretch(1)
        layout.addLayout(playback_layout)

        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶ Start Rally")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self._on_start_rally)
        controls_layout.addWidget(self.start_btn)

        self.end_btn = QPushButton("⏹ End Rally")
        self.end_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px;")
        self.end_btn.setEnabled(False)
        self.end_btn.clicked.connect(self._on_end_rally)
        controls_layout.addWidget(self.end_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; padding: 8px;")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_rally)
        controls_layout.addWidget(self.cancel_btn)

        layout.addLayout(controls_layout)

        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet("color: #ffc107; font-style: italic;")
        self.pending_label.setVisible(False)
        layout.addWidget(self.pending_label)

        action_edit_layout = QHBoxLayout()
        action_edit_layout.setContentsMargins(0, 2, 0, 4)
        action_edit_layout.setSpacing(6)

        self.segment_class_combo = QComboBox()
        for action_type in self.action_types:
            self.segment_class_combo.addItem(action_type)
        self.segment_class_combo.setEnabled(False)
        action_edit_layout.addWidget(self.segment_class_combo, stretch=1)

        self.apply_segment_btn = QPushButton("Apply To Action")
        self.apply_segment_btn.setEnabled(False)
        self.apply_segment_btn.clicked.connect(self._apply_action_reclass)
        action_edit_layout.addWidget(self.apply_segment_btn)

        layout.addLayout(action_edit_layout)

        self.segment_hint_label = QLabel("")
        self.segment_hint_label.setStyleSheet("color: #8fa1b7; font-size: 11px;")
        self.segment_hint_label.setVisible(False)
        layout.addWidget(self.segment_hint_label)

        list_header = QHBoxLayout()
        list_label = QLabel("Rallies")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        list_header.addWidget(list_label)

        self.delete_btn = QPushButton("Delete Rally")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected_rally)
        list_header.addWidget(self.delete_btn)
        layout.addLayout(list_header)

        self.rally_tree = QTreeWidget()
        self.rally_tree.setHeaderHidden(True)
        self.rally_tree.setIndentation(24)
        self.rally_tree.setRootIsDecorated(True)
        self.rally_tree.setUniformRowHeights(False)
        self.rally_tree.setStyleSheet(
            "QTreeWidget {"
            "background-color: #15191f;"
            "border: 1px solid #2e3642;"
            "border-radius: 12px;"
            "padding: 6px;"
            "outline: none;"
            "font-size: 13px;"
            "}"
            "QTreeWidget::item {"
            "padding: 7px 8px;"
            "margin: 2px 0;"
            "border-radius: 8px;"
            "}"
            "QTreeWidget::item:selected {"
            "background-color: #254a78;"
            "color: white;"
            "}"
            "QTreeWidget::item:hover {"
            "background-color: #1d2630;"
            "}"
        )
        self.rally_tree.itemClicked.connect(self._on_item_clicked)
        self.rally_tree.currentItemChanged.connect(self._on_current_item_changed)
        self.rally_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.rally_tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        layout.addWidget(self.rally_tree)

        self.stats_label = QLabel("Total Rallies: 0")
        self.stats_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.stats_label)

    def _on_start_rally(self):
        self.start_rally_requested.emit()

    def _on_end_rally(self):
        self.end_rally_requested.emit()

    def _on_cancel_rally(self):
        self.cancel_rally_requested.emit()

    def set_pending_rally(self, is_pending: bool, start_frame: int | None = None):
        """Sync pending-rally controls with annotation state."""
        self.is_rally_pending = is_pending
        self.start_btn.setEnabled(not is_pending)
        self.end_btn.setEnabled(is_pending)
        self.cancel_btn.setEnabled(is_pending)
        if is_pending:
            label = "Recording rally"
            if start_frame is not None:
                label = f"Recording rally from frame {start_frame}"
            self.pending_label.setText(label)
            self.pending_label.setVisible(True)
        else:
            self.pending_label.clear()
            self.pending_label.setVisible(False)

    def cancel_pending_rally(self):
        self.set_pending_rally(False)

    def set_current_context(self, current_frame: int, focus_rally_id: int | None = None):
        """Update frame context used to highlight the current action inside the tree."""
        self.current_frame = int(current_frame)
        self.focus_rally_id = focus_rally_id
        self._refresh_list()

    def _select_action_type(self, action_type: str):
        self.selected_action_type = action_type
        self._update_action_buttons()
        self.action_type_selected.emit(action_type)

    def _update_action_buttons(self):
        for action_type, button in self.action_buttons.items():
            is_selected = action_type == self.selected_action_type
            button.setChecked(is_selected)
            if is_selected:
                button.setStyleSheet(
                    "QPushButton {"
                    "background-color: #2d6cdf; color: white; font-weight: bold;"
                    "border: 1px solid #74a7ff; border-radius: 8px; padding: 4px 8px;"
                    "text-align: center; font-size: 11px; }"
                )
            else:
                button.setStyleSheet(
                    "QPushButton {"
                    "background-color: #2b3038; color: #d0d7de;"
                    "border: 1px solid #3b4250; border-radius: 8px; padding: 4px 8px;"
                    "text-align: center; font-size: 11px; }"
                    "QPushButton:hover { background-color: #343b45; }"
                )

        if self.selected_action_type:
            self.selected_type_label.setText(f"Текущее действие: {self.selected_action_type}")

    def set_rallies(self, rallies: List[Dict], pending_rally: Dict | None = None):
        self.rallies = sorted(
            rallies,
            key=lambda rally: (
                int(rally.get("start_frame", 0)),
                int(rally.get("end_frame", 0)),
                int(rally.get("id", 0)),
            ),
        )
        self.pending_rally = pending_rally
        self._refresh_list()

    def _refresh_list(self):
        current_rally_id = self._selected_rally_id()
        self.rally_tree.clear()
        self.selected_action_item = None

        annotated_rallies = 0
        visible_rallies: List[Dict] = list(self.rallies)
        if self.pending_rally is not None:
            visible_rallies.append(self.pending_rally)
        pending_item = None

        for rally in visible_rallies:
            start_frame = int(rally.get("start_frame", 0))
            end_frame = int(rally.get("end_frame", 0))
            has_actions = bool(rally.get("actions"))
            if has_actions and not rally.get("is_pending"):
                annotated_rallies += 1

            title = f"Rally #{rally['id']}   {start_frame}-{end_frame}"
            if rally.get("is_pending"):
                title = f"Rally [open]   {start_frame}-{end_frame}"

            item = QTreeWidgetItem([title])
            item.setData(0, Qt.ItemDataRole.UserRole, int(rally["id"]))
            item.setData(0, Qt.ItemDataRole.UserRole + 1, int(rally["id"]))
            item.setData(0, Qt.ItemDataRole.UserRole + 2, start_frame)
            item.setData(0, Qt.ItemDataRole.UserRole + 3, bool(not rally.get("is_pending")))
            item.setToolTip(0, f"Frames {start_frame}-{end_frame}")
            self._style_tree_item(item, "pending_rally" if rally.get("is_pending") else "rally")
            self.rally_tree.addTopLevelItem(item)
            if rally.get("is_pending"):
                pending_item = item

            children = rally.get("hierarchy") or self._fallback_hierarchy(rally)
            for child in children:
                child_item = QTreeWidgetItem([self._format_hierarchy_label(child)])
                child_item.setData(0, Qt.ItemDataRole.UserRole, int(rally["id"]))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 1, int(rally["id"]))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 2, self._target_frame(child))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 3, bool(not rally.get("is_pending")))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 4, str(child.get("kind", "")))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 5, int(child.get("start_frame", -1)))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 6, int(child.get("end_frame", -1)))
                child_item.setData(0, Qt.ItemDataRole.UserRole + 7, str(child.get("type", "")))
                child_item.setToolTip(0, self._format_hierarchy_tooltip(child))
                child_kind = str(child.get("kind", "action"))
                is_current_action = (
                    child_kind == "action"
                    and self._is_focus_rally(int(rally["id"]))
                    and int(child.get("start_frame", -1)) <= self.current_frame <= int(child.get("end_frame", -1))
                )
                self._style_tree_item(
                    child_item,
                    "current_action" if is_current_action else child_kind,
                )
                item.addChild(child_item)

            item.setExpanded(True)
            if current_rally_id == rally["id"]:
                self.rally_tree.setCurrentItem(item)

        if pending_item is not None:
            self.rally_tree.setCurrentItem(pending_item)
            self.rally_tree.scrollToItem(pending_item, QAbstractItemView.ScrollHint.PositionAtBottom)

        selected_item = self.rally_tree.currentItem()
        can_delete = bool(selected_item and selected_item.data(0, Qt.ItemDataRole.UserRole + 3))
        self.delete_btn.setEnabled(can_delete)
        self._sync_action_editor(selected_item)
        open_count = 1 if self.pending_rally is not None else 0
        self.stats_label.setText(
            f"Total Rallies: {len(self.rallies)} | Annotated: {annotated_rallies} | Open: {open_count}"
        )

    def _selected_rally_id(self) -> int:
        item = self.rally_tree.currentItem()
        if item is None:
            return -1
        return int(item.data(0, Qt.ItemDataRole.UserRole))

    def _on_current_item_changed(self, current, previous):
        del previous
        rally_id = self._selected_rally_id()
        can_delete = bool(current and current.data(0, Qt.ItemDataRole.UserRole + 3))
        self.delete_btn.setEnabled(can_delete)
        self._sync_action_editor(current)
        self.rally_selected.emit(rally_id)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        del column
        rally_id = int(item.data(0, Qt.ItemDataRole.UserRole))
        target_frame = item.data(0, Qt.ItemDataRole.UserRole + 2)
        self.rally_selected.emit(rally_id)
        if target_frame is not None:
            self.seek_to_rally.emit(int(target_frame))

    def _delete_selected_rally(self):
        rally_id = self._selected_rally_id()
        if rally_id < 0:
            return
        self.rally_deleted.emit(rally_id)

    def _show_tree_context_menu(self, pos):
        item = self.rally_tree.itemAt(pos)
        if item is None:
            return

        if str(item.data(0, Qt.ItemDataRole.UserRole + 4)) != "action":
            return

        self.rally_tree.setCurrentItem(item)
        action_item = {
            "rally_id": int(item.data(0, Qt.ItemDataRole.UserRole)),
            "start_frame": int(item.data(0, Qt.ItemDataRole.UserRole + 5)),
            "end_frame": int(item.data(0, Qt.ItemDataRole.UserRole + 6)),
            "type": str(item.data(0, Qt.ItemDataRole.UserRole + 7)),
        }

        menu = QMenu(self)
        change_menu = menu.addMenu("Change To")
        for action_type in self.action_types:
            action = change_menu.addAction(action_type)
            action.triggered.connect(
                lambda checked=False, value=action_type, payload=action_item: self.action_reclass_requested.emit(
                    payload["rally_id"],
                    payload["start_frame"],
                    payload["end_frame"],
                    payload["type"],
                    value,
                )
            )

        menu.exec(self.rally_tree.viewport().mapToGlobal(pos))

    def _apply_action_reclass(self):
        if not self.selected_action_item:
            return
        self.action_reclass_requested.emit(
            int(self.selected_action_item["rally_id"]),
            int(self.selected_action_item["start_frame"]),
            int(self.selected_action_item["end_frame"]),
            str(self.selected_action_item["type"]),
            self.segment_class_combo.currentText(),
        )

    def _sync_action_editor(self, item: QTreeWidgetItem | None):
        self.selected_action_item = None
        if item is not None and str(item.data(0, Qt.ItemDataRole.UserRole + 4)) == "action":
            action_type = str(item.data(0, Qt.ItemDataRole.UserRole + 7))
            self.selected_action_item = {
                "rally_id": int(item.data(0, Qt.ItemDataRole.UserRole)),
                "start_frame": int(item.data(0, Qt.ItemDataRole.UserRole + 5)),
                "end_frame": int(item.data(0, Qt.ItemDataRole.UserRole + 6)),
                "type": action_type,
            }
            combo_index = self.segment_class_combo.findText(action_type)
            if combo_index >= 0:
                self.segment_class_combo.setCurrentIndex(combo_index)
            self.segment_hint_label.setText(
                f"Selected action: {action_type}   "
                f"{self.selected_action_item['start_frame']}-{self.selected_action_item['end_frame']}"
            )
            self.segment_hint_label.setVisible(True)
            self.segment_class_combo.setEnabled(True)
            self.apply_segment_btn.setEnabled(True)
            return

        self.segment_hint_label.clear()
        self.segment_hint_label.setVisible(False)
        self.segment_class_combo.setEnabled(False)
        self.apply_segment_btn.setEnabled(False)

    def _is_focus_rally(self, rally_id: int) -> bool:
        if self.focus_rally_id is None:
            return True
        return int(rally_id) == int(self.focus_rally_id)

    @staticmethod
    def _target_frame(item: Dict) -> int:
        if "frame" in item:
            return int(item["frame"])
        return int(item.get("start_frame", 0))

    @staticmethod
    def _format_hierarchy_label(item: Dict) -> str:
        item_type = str(item.get("type", ""))
        if item.get("kind") == "marker":
            frame = int(item.get("frame", 0))
            label = (
                "Start" if item_type == "RallyStart"
                else "End" if item_type == "RallyEnd"
                else "Current" if item_type == "CurrentFrame"
                else item_type
            )
            return f"{label}   [{frame}]"

        start_frame = int(item.get("start_frame", 0))
        end_frame = int(item.get("end_frame", start_frame))
        return f"{item_type}   {start_frame}-{end_frame}"

    @staticmethod
    def _format_hierarchy_tooltip(item: Dict) -> str:
        if item.get("kind") == "marker":
            return f"Frame {int(item.get('frame', 0))}"
        return f"Frames {int(item.get('start_frame', 0))}-{int(item.get('end_frame', 0))}"

    @staticmethod
    def _fallback_hierarchy(rally: Dict) -> List[Dict]:
        hierarchy = [
            {
                "kind": "marker",
                "type": "RallyStart",
                "frame": int(rally.get("start_frame", 0)),
            }
        ]
        hierarchy.extend(rally.get("actions", []))
        hierarchy.append(
            {
                "kind": "marker",
                "type": "RallyEnd",
                "frame": int(rally.get("end_frame", 0)),
            }
        )
        return hierarchy

    @staticmethod
    def _style_tree_item(item: QTreeWidgetItem, item_kind: str):
        if item_kind == "rally":
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#f3f6fb")))
            item.setBackground(0, QBrush(QColor("#202833")))
            item.setSizeHint(0, QSize(0, 38))
            return

        if item_kind == "pending_rally":
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#fff3bf")))
            item.setBackground(0, QBrush(QColor("#3b3320")))
            item.setSizeHint(0, QSize(0, 38))
            return

        if item_kind == "marker":
            font = QFont()
            font.setPointSize(11)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(QColor("#8fa1b7")))
            item.setSizeHint(0, QSize(0, 28))
            return

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor("#8bd3ff")))
        item.setBackground(0, QBrush(QColor("#18222c")))
        item.setSizeHint(0, QSize(0, 32))

        if item_kind == "current_action":
            item.setForeground(0, QBrush(QColor("#fff7cc")))
            item.setBackground(0, QBrush(QColor("#5b4621")))
