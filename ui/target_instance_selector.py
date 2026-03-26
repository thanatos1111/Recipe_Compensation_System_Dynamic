"""
Target instance selector panel (Milestone 2).

Shows Target ID options for the selected material.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QComboBox, QVBoxLayout, QWidget


class TargetInstanceSelectorPanel(QWidget):
    targetSelected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._targets: list[str] = []

        self.material_label = QLabel("Targets:")
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Target instance selector (Milestone 2)"))
        layout.addWidget(self.material_label)
        layout.addWidget(self.target_combo)
        self.setLayout(layout)

        self.target_combo.currentTextChanged.connect(self._emit_target_selected)

    def set_targets(self, target_ids: list[str]) -> None:
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self._targets = list(target_ids)
        if target_ids:
            self.target_combo.addItems(target_ids)
            self.target_combo.setEnabled(True)
            self.target_combo.setCurrentIndex(0)
        else:
            self.target_combo.setEnabled(False)
        self.target_combo.blockSignals(False)

    def _emit_target_selected(self, target_id: str) -> None:
        if target_id:
            self.targetSelected.emit(target_id)

