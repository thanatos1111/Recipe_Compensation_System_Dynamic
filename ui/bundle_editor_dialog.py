from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from core.bundles import validate_custom_bundle
from core.model_registry import get_supported_model_specs


class BundleEditorDialog(QDialog):
    """
    Simple editor for a custom bundle (name + rs/thickness/rsu model dropdowns).
    """

    def __init__(
        self,
        *,
        title: str,
        preset_bundle_names: set[str],
        initial_name: str = "",
        initial_models: Optional[Mapping[str, Any]] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        supported = list(get_supported_model_specs().keys())
        supported.sort()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. my_fast_bundle")
        self.name_edit.setText(str(initial_name or ""))

        self.rs_combo = QComboBox()
        self.rs_combo.addItems(supported)
        self.thickness_combo = QComboBox()
        self.thickness_combo.addItems(supported)
        self.rsu_combo = QComboBox()
        self.rsu_combo.addItems(supported)

        init = dict(initial_models or {})
        self._set_combo_value(self.rs_combo, str(init.get("rs") or "gbr"))
        self._set_combo_value(self.thickness_combo, str(init.get("thickness") or "gbr"))
        self._set_combo_value(self.rsu_combo, str(init.get("rsu") or "gbr"))

        self._preset_bundle_names = set(preset_bundle_names or set())

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Bundle name:", self.name_edit)
        form.addRow("RS model:", self.rs_combo)
        form.addRow("Thickness model:", self.thickness_combo)
        form.addRow("RSU model:", self.rsu_combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        v = str(value or "").strip()
        if not v:
            return
        idx = combo.findText(v)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def value(self) -> tuple[str, dict[str, str]]:
        name = str(self.name_edit.text() or "").strip()
        models = {
            "rs": str(self.rs_combo.currentText() or "").strip(),
            "thickness": str(self.thickness_combo.currentText() or "").strip(),
            "rsu": str(self.rsu_combo.currentText() or "").strip(),
        }
        return name, models

    def _on_accept(self) -> None:
        name, models = self.value()
        try:
            validate_custom_bundle(name=name, models=models, preset_names=self._preset_bundle_names)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid bundle", str(exc))
            return
        self.accept()

