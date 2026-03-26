"""
Material selector panel (Milestone 2).

Includes workbook open action and a material dropdown.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MaterialSelectorPanel(QWidget):
    materialSelected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._on_open_workbook: Optional[Callable[[str], None]] = None

        self.open_button = QPushButton("Open Workbook (.xlsx)")
        self.material_combo = QComboBox()
        self.material_combo.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Material selector (Milestone 2)"))
        layout.addWidget(self.open_button)
        layout.addWidget(QLabel("Materials:"))
        layout.addWidget(self.material_combo)

        self.setLayout(layout)

        self.open_button.clicked.connect(self._handle_open_clicked)
        self.material_combo.currentTextChanged.connect(self._emit_material_selected)

    def set_on_open_workbook(self, callback: Callable[[str], None]) -> None:
        self._on_open_workbook = callback

    def set_materials(self, materials: list[str]) -> None:
        self.material_combo.blockSignals(True)
        self.material_combo.clear()
        if materials:
            self.material_combo.addItems(materials)
            self.material_combo.setEnabled(True)
            self.material_combo.setCurrentIndex(0)
        else:
            self.material_combo.setEnabled(False)
        self.material_combo.blockSignals(False)

    def _handle_open_clicked(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open Excel Workbook", "", "Excel Files (*.xlsx)")
        if not path:
            return
        if self._on_open_workbook is not None:
            self._on_open_workbook(path)

    def _emit_material_selected(self, material_name: str) -> None:
        if material_name:
            self.materialSelected.emit(material_name)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Material selector (placeholder)."))
        self.setLayout(layout)

