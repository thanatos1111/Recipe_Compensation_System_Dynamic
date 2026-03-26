"""
Recommendation panel (placeholder).
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class RecommendationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Recommendation panel (placeholder)."))
        self.setLayout(layout)

