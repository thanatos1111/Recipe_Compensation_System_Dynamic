"""
Model training/metrics panel (placeholder).
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ModelPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model panel (placeholder)."))
        self.setLayout(layout)

