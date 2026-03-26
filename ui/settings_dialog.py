"""
Settings dialog (Milestone 4 add-on).

Edits parameter constraints JSON and persists to user config.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        initial_value: dict[str, Any],
        on_save: Callable[[dict[str, Any]], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._on_save = on_save

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Edit JSON and click Save."))

        self.editor = QTextEdit()
        self.editor.setPlainText(json.dumps(initial_value, indent=2, ensure_ascii=False))
        layout.addWidget(self.editor)

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
        raw = self.editor.toPlainText()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Root must be a JSON object.")
        except Exception as e:
            QMessageBox.critical(self, "Invalid JSON", str(e))
            return

        self._on_save(data)
        self.accept()

