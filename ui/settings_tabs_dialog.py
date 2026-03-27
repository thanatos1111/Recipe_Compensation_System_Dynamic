"""
Tabbed settings dialog (Milestone 6 usability add-on).

Tab 1: parameter constraints JSON
Tab 2: minimum steps JSON
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)


class SettingsTabsDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        initial_parameter_constraints: dict[str, Any],
        initial_minimum_steps: dict[str, Any],
        on_save: Callable[[dict[str, Any], dict[str, Any]], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._on_save = on_save

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Edit settings. JSON must be valid objects."))

        self.tabs = QTabWidget()

        # Parameter constraints tab
        self.constraints_editor = QTextEdit()
        self.constraints_editor.setPlainText(
            json.dumps(initial_parameter_constraints or {}, indent=2, ensure_ascii=False)
        )
        tab1 = QVBoxLayout()
        tab1.addWidget(QLabel("parameter_constraints"))
        tab1.addWidget(self.constraints_editor)
        constraints_widget = QDialog()
        constraints_widget.setLayout(tab1)

        # Minimum steps tab
        self.min_steps_editor = QTextEdit()
        self.min_steps_editor.setPlainText(json.dumps(initial_minimum_steps or {}, indent=2, ensure_ascii=False))
        tab2 = QVBoxLayout()
        tab2.addWidget(QLabel("minimum_steps"))
        tab2.addWidget(self.min_steps_editor)
        min_steps_widget = QDialog()
        min_steps_widget.setLayout(tab2)

        self.tabs.addTab(constraints_widget, "Constraints")
        self.tabs.addTab(min_steps_widget, "Minimum Steps")
        layout.addWidget(self.tabs)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

        self.save_button.clicked.connect(self._save_clicked)
        self.cancel_button.clicked.connect(self.reject)

    def _save_clicked(self) -> None:
        try:
            constraints = json.loads(self.constraints_editor.toPlainText())
            min_steps = json.loads(self.min_steps_editor.toPlainText())
            if not isinstance(constraints, dict):
                raise ValueError("parameter_constraints must be a JSON object")
            if not isinstance(min_steps, dict):
                raise ValueError("minimum_steps must be a JSON object")
        except Exception as e:
            QMessageBox.critical(self, "Invalid JSON", str(e))
            return

        self._on_save(constraints, min_steps)
        self.accept()

