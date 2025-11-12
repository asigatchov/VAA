"""Action panel widget for managing action annotations."""
from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel,
    QComboBox, QHeaderView, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction


class ActionPanel(QWidget):
    """Panel for action type selection and action list management."""
    
    # Signals
    start_action_requested = pyqtSignal(str)  # Emits action type
    end_action_requested = pyqtSignal()
    action_deleted = pyqtSignal(int)  # Emits action ID
    action_selected = pyqtSignal(int)  # Emits action ID
    seek_to_action = pyqtSignal(int)  # Emits frame index
    
    def __init__(self, action_types: List[str], parent=None):
        super().__init__(parent)
        self.action_types = action_types
        self.actions: List[Dict] = []
        self.is_action_pending = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Action type selector
        type_layout = QVBoxLayout()
        type_label = QLabel("Action Type:")
        type_label.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.action_types)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Action controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Action")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self._on_start_action)
        controls_layout.addWidget(self.start_btn)
        
        self.end_btn = QPushButton("⏹ End Action")
        self.end_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px;")
        self.end_btn.setEnabled(False)
        self.end_btn.clicked.connect(self._on_end_action)
        controls_layout.addWidget(self.end_btn)
        
        layout.addLayout(controls_layout)
        
        # Pending action indicator
        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet("color: #ffc107; font-style: italic;")
        self.pending_label.setVisible(False)
        layout.addWidget(self.pending_label)
        
        # Action list table
        list_label = QLabel("Actions:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.action_table = QTableWidget()
        self.action_table.setColumnCount(4)
        self.action_table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Type", "Duration (s)"])
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.action_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.action_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.action_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.action_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.action_table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.action_table)
        
        # Statistics
        self.stats_label = QLabel("Total Actions: 0")
        self.stats_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.stats_label)
    
    def _on_start_action(self):
        """Handle start action button click."""
        action_type = self.type_combo.currentText()
        self.is_action_pending = True
        self.start_btn.setEnabled(False)
        self.end_btn.setEnabled(True)
        self.pending_label.setText(f"⏺ Recording: {action_type}")
        self.pending_label.setVisible(True)
        self.start_action_requested.emit(action_type)
    
    def _on_end_action(self):
        """Handle end action button click."""
        self.is_action_pending = False
        self.start_btn.setEnabled(True)
        self.end_btn.setEnabled(False)
        self.pending_label.setVisible(False)
        self.end_action_requested.emit()
    
    def cancel_pending_action(self):
        """Cancel the current pending action."""
        if self.is_action_pending:
            self.is_action_pending = False
            self.start_btn.setEnabled(True)
            self.end_btn.setEnabled(False)
            self.pending_label.setVisible(False)
    
    def set_actions(self, actions: List[Dict]):
        """Update the action list."""
        self.actions = actions
        self._refresh_table()
    
    def _refresh_table(self):
        """Refresh the action table display."""
        self.action_table.setRowCount(len(self.actions))
        
        for row, action in enumerate(self.actions):
            start_time = action.get("start_time", 0.0)
            end_time = action.get("end_time", 0.0)
            action_type = action.get("type", "")
            duration = end_time - start_time
            
            self.action_table.setItem(row, 0, QTableWidgetItem(f"{start_time:.2f}"))
            self.action_table.setItem(row, 1, QTableWidgetItem(f"{end_time:.2f}"))
            self.action_table.setItem(row, 2, QTableWidgetItem(action_type))
            self.action_table.setItem(row, 3, QTableWidgetItem(f"{duration:.2f}"))
        
        self.stats_label.setText(f"Total Actions: {len(self.actions)}")
    
    def _on_selection_changed(self):
        """Handle table row selection."""
        selected_rows = self.action_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.actions):
                action_id = self.actions[row].get("id", -1)
                self.action_selected.emit(action_id)
    
    def _show_context_menu(self, pos):
        """Show context menu for action table."""
        item = self.action_table.itemAt(pos)
        if item is None:
            return
        
        row = item.row()
        if row < 0 or row >= len(self.actions):
            return
        
        action = self.actions[row]
        
        menu = QMenu(self)
        
        goto_start_action = QAction("Go to Start", self)
        goto_start_action.triggered.connect(lambda: self.seek_to_action.emit(action["start_frame"]))
        menu.addAction(goto_start_action)
        
        goto_end_action = QAction("Go to End", self)
        goto_end_action.triggered.connect(lambda: self.seek_to_action.emit(action["end_frame"]))
        menu.addAction(goto_end_action)
        
        menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.action_deleted.emit(action["id"]))
        menu.addAction(delete_action)
        
        menu.exec(self.action_table.mapToGlobal(pos))
