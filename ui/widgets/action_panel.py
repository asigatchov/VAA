"""Action panel widget for managing action annotations."""
from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel,
    QHeaderView, QMenu, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction


class ActionPanel(QWidget):
    """Panel for rally management and step selection."""
    
    # Signals
    start_rally_requested = pyqtSignal()
    end_rally_requested = pyqtSignal()
    action_type_selected = pyqtSignal(str)  # Emits selected action type
    rally_deleted = pyqtSignal(int)  # Emits rally ID
    rally_selected = pyqtSignal(int)  # Emits rally ID
    seek_to_rally = pyqtSignal(int)  # Emits frame index
    
    def __init__(self, action_types: List[str], parent=None):
        super().__init__(parent)
        self.action_types = action_types
        self.rallies: List[Dict] = []
        self.table_rows: List[Dict] = []
        self.is_rally_pending = False
        self.selected_action_type = self.action_types[0] if self.action_types else ""
        self.action_buttons: Dict[str, QPushButton] = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        selector_label = QLabel("Rally Flow")
        selector_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(selector_label)

        selector_hint = QLabel("Выберите шаг розыгрыша перед началом разметки.")
        selector_hint.setWordWrap(True)
        selector_hint.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        layout.addWidget(selector_hint)

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
            button.clicked.connect(lambda checked, value=action_type: self._select_action_type(value))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(28)
            button.setMinimumWidth(68)
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
        
        # Rally controls
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
        
        layout.addLayout(controls_layout)
        
        # Pending rally indicator
        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet("color: #ffc107; font-style: italic;")
        self.pending_label.setVisible(False)
        layout.addWidget(self.pending_label)
        
        list_label = QLabel("Rallies:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.action_table = QTableWidget()
        self.action_table.setColumnCount(5)
        self.action_table.setHorizontalHeaderLabels(["Level", "Label", "Start", "End", "Frames"])
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.action_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.action_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.action_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.action_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.action_table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.action_table)
        
        # Statistics
        self.stats_label = QLabel("Total Rallies: 0")
        self.stats_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.stats_label)
    
    def _on_start_rally(self):
        """Handle start rally button click."""
        self.is_rally_pending = True
        self.start_btn.setEnabled(False)
        self.end_btn.setEnabled(True)
        self.pending_label.setText("⏺ Recording rally")
        self.pending_label.setVisible(True)
        self.start_rally_requested.emit()
    
    def _on_end_rally(self):
        """Handle end rally button click."""
        self.is_rally_pending = False
        self.start_btn.setEnabled(True)
        self.end_btn.setEnabled(False)
        self.pending_label.setVisible(False)
        self.end_rally_requested.emit()
    
    def cancel_pending_rally(self):
        """Cancel the current pending rally."""
        if self.is_rally_pending:
            self.is_rally_pending = False
            self.start_btn.setEnabled(True)
            self.end_btn.setEnabled(False)
            self.pending_label.setVisible(False)

    def _select_action_type(self, action_type: str):
        """Select current action type from rally flow buttons."""
        self.selected_action_type = action_type
        self._update_action_buttons()
        self.action_type_selected.emit(action_type)

    def _update_action_buttons(self):
        """Refresh button styles for the selected action type."""
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
    
    def set_rallies(self, rallies: List[Dict]):
        """Update the rally list."""
        self.rallies = rallies
        self._refresh_table()
    
    def _refresh_table(self):
        """Refresh the rally table display."""
        self.table_rows = []
        derived_actions = 0
        for rally in self.rallies:
            rally_actions = rally.get("actions", [])
            derived_actions += len(rally_actions)
            self.table_rows.append({
                "kind": "rally",
                "rally_id": rally["id"],
                "label": f"Rally #{rally['id']}",
                "start_time": rally.get("start_time", 0.0),
                "end_time": rally.get("end_time", 0.0),
                "start_frame": rally.get("start_frame", 0),
                "end_frame": rally.get("end_frame", 0),
            })
            for child in rally.get("hierarchy", []):
                self.table_rows.append(dict(child))

        self.action_table.setRowCount(len(self.table_rows))
        for row, item in enumerate(self.table_rows):
            kind = item.get("kind", "")
            if kind == "rally":
                level = "rally"
                label = item["label"]
                start_value = f"{item.get('start_time', 0.0):.2f}"
                end_value = f"{item.get('end_time', 0.0):.2f}"
                frames_value = f"{item.get('start_frame', 0)}-{item.get('end_frame', 0)}"
            elif kind == "marker":
                level = "  marker"
                label = item.get("label", "")
                start_value = f"{item.get('time', 0.0):.2f}"
                end_value = "-"
                frames_value = str(item.get("frame", 0))
            else:
                level = "  action"
                label = item.get("label", "")
                start_value = f"{item.get('start_time', 0.0):.2f}"
                end_value = f"{item.get('end_time', 0.0):.2f}"
                frames_value = f"{item.get('start_frame', 0)}-{item.get('end_frame', 0)}"

            self.action_table.setItem(row, 0, QTableWidgetItem(level))
            self.action_table.setItem(row, 1, QTableWidgetItem(label))
            self.action_table.setItem(row, 2, QTableWidgetItem(start_value))
            self.action_table.setItem(row, 3, QTableWidgetItem(end_value))
            self.action_table.setItem(row, 4, QTableWidgetItem(frames_value))
        
        self.stats_label.setText(f"Total Rallies: {len(self.rallies)} | Derived Actions: {derived_actions}")
    
    def _on_selection_changed(self):
        """Handle table row selection."""
        selected_rows = self.action_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.table_rows):
                rally_id = self.table_rows[row].get("rally_id", -1)
                self.rally_selected.emit(rally_id)
    
    def _show_context_menu(self, pos):
        """Show context menu for action table."""
        item = self.action_table.itemAt(pos)
        if item is None:
            return
        
        row = item.row()
        if row < 0 or row >= len(self.table_rows):
            return
        
        item_data = self.table_rows[row]
        rally_id = item_data.get("rally_id", -1)
        rally = next((item for item in self.rallies if item.get("id") == rally_id), None)
        if rally is None:
            return
        
        menu = QMenu(self)
        
        goto_start_action = QAction("Go to Start", self)
        target_start = item_data.get("frame", item_data.get("start_frame", rally["start_frame"]))
        goto_start_action.triggered.connect(lambda: self.seek_to_rally.emit(target_start))
        menu.addAction(goto_start_action)
        
        goto_end_action = QAction("Go to End", self)
        target_end = item_data.get("frame", item_data.get("end_frame", rally["end_frame"]))
        goto_end_action.triggered.connect(lambda: self.seek_to_rally.emit(target_end))
        menu.addAction(goto_end_action)
        
        menu.addSeparator()
        
        delete_rally = QAction("Delete Rally", self)
        delete_rally.triggered.connect(lambda: self.rally_deleted.emit(rally["id"]))
        menu.addAction(delete_rally)
        
        menu.exec(self.action_table.mapToGlobal(pos))
