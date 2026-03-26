"""
Spec configuration panel (Milestone 3).

Allows users to configure RS/RSU/thickness spec modes and apply them to the
currently selected material.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from core.schemas import SpecConfig


class SpecConfigPanel(QWidget):
    specApplied = Signal(object)  # emits SpecConfig

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Spec config (Milestone 3)"))

        self.save_per_target_checkbox = QCheckBox("Save/apply spec for active Target ID only (otherwise per material)")
        self.save_per_target_checkbox.setChecked(True)
        layout.addWidget(self.save_per_target_checkbox)

        self.max_lifetime_checkbox = QCheckBox("Enable max lifetime (kW*h) for this material/target")
        self.max_lifetime_checkbox.setChecked(False)
        layout.addWidget(self.max_lifetime_checkbox)

        max_lt_row = QHBoxLayout()
        max_lt_row.addWidget(QLabel("Max lifetime"))
        self.max_lifetime_spin = QDoubleSpinBox()
        self.max_lifetime_spin.setRange(0.0, 1e9)
        self.max_lifetime_spin.setDecimals(6)
        self.max_lifetime_spin.setValue(0.0)
        max_lt_row.addWidget(self.max_lifetime_spin)
        layout.addLayout(max_lt_row)

        # Enable/disable individual spec components.
        self.rs_enabled_checkbox = QCheckBox("Enable RS spec filtering")
        self.rs_enabled_checkbox.setChecked(True)
        self.th_enabled_checkbox = QCheckBox("Enable Thickness spec filtering")
        self.th_enabled_checkbox.setChecked(True)

        # RS group
        rs_group = QGroupBox("RS spec")
        rs_layout = QVBoxLayout()

        self.rs_target_radio = QRadioButton("Target mode (RS target + tolerance)")
        self.rs_range_radio = QRadioButton("Range mode (RS min/max)")
        rs_layout.addWidget(self.rs_target_radio)
        rs_layout.addWidget(self.rs_range_radio)

        rs_target_row = QHBoxLayout()
        rs_target_row.addWidget(QLabel("RS target"))
        self.rs_target_spin = QDoubleSpinBox()
        self.rs_target_spin.setRange(-1.0, 1e6)
        self.rs_target_spin.setDecimals(6)
        self.rs_target_spin.setValue(-1.0)  # -1 means "unset"
        rs_target_row.addWidget(self.rs_target_spin)

        rs_tol_row = QHBoxLayout()
        rs_tol_row.addWidget(QLabel("RS tol"))
        self.rs_tol_spin = QDoubleSpinBox()
        self.rs_tol_spin.setRange(-1.0, 1e6)
        self.rs_tol_spin.setDecimals(6)
        self.rs_tol_spin.setValue(-1.0)
        rs_tol_row.addWidget(self.rs_tol_spin)

        rs_range_target = QHBoxLayout()
        rs_range_target.addWidget(QLabel("RS min"))
        self.rs_min_spin = QDoubleSpinBox()
        self.rs_min_spin.setRange(-1.0, 1e6)
        self.rs_min_spin.setDecimals(6)
        self.rs_min_spin.setValue(-1.0)
        rs_range_target.addWidget(self.rs_min_spin)

        rs_range_target_max = QHBoxLayout()
        rs_range_target_max.addWidget(QLabel("RS max"))
        self.rs_max_spin = QDoubleSpinBox()
        self.rs_max_spin.setRange(-1.0, 1e6)
        self.rs_max_spin.setDecimals(6)
        self.rs_max_spin.setValue(-1.0)
        rs_range_target_max.addWidget(self.rs_max_spin)

        rs_layout.addLayout(rs_target_row)
        rs_layout.addLayout(rs_tol_row)
        rs_layout.addLayout(rs_range_target)
        rs_layout.addLayout(rs_range_target_max)
        rs_layout.insertWidget(0, self.rs_enabled_checkbox)

        # RSU group
        rsu_group = QGroupBox("RSU spec")
        rsu_layout = QVBoxLayout()

        self.rsu_enabled_checkbox = QCheckBox("Enable RSU spec filtering (max threshold)")
        self.rsu_enabled_checkbox.setChecked(False)
        rsu_layout.addWidget(self.rsu_enabled_checkbox)

        rsu_row = QHBoxLayout()
        rsu_row.addWidget(QLabel("RSU max"))
        self.rsu_max_spin = QDoubleSpinBox()
        self.rsu_max_spin.setRange(0.0, 1e6)
        self.rsu_max_spin.setDecimals(6)
        self.rsu_max_spin.setValue(0.0)
        rsu_row.addWidget(self.rsu_max_spin)
        rsu_layout.addLayout(rsu_row)

        # Thickness group
        th_group = QGroupBox("Thickness spec")
        th_layout = QVBoxLayout()

        self.th_target_radio = QRadioButton("Target mode (thickness target + tol)")
        self.th_range_radio = QRadioButton("Range mode (thickness min/max)")
        th_layout.addWidget(self.th_target_radio)
        th_layout.addWidget(self.th_range_radio)

        th_target_row = QHBoxLayout()
        th_target_row.addWidget(QLabel("Thickness target"))
        self.th_target_spin = QDoubleSpinBox()
        self.th_target_spin.setRange(-1.0, 1e6)
        self.th_target_spin.setDecimals(6)
        self.th_target_spin.setValue(-1.0)
        th_target_row.addWidget(self.th_target_spin)

        th_tol_row = QHBoxLayout()
        th_tol_row.addWidget(QLabel("Thickness tol"))
        self.th_tol_spin = QDoubleSpinBox()
        self.th_tol_spin.setRange(-1.0, 1e6)
        self.th_tol_spin.setDecimals(6)
        self.th_tol_spin.setValue(-1.0)
        th_tol_row.addWidget(self.th_tol_spin)

        th_range_min_row = QHBoxLayout()
        th_range_min_row.addWidget(QLabel("Thickness min"))
        self.th_min_spin = QDoubleSpinBox()
        self.th_min_spin.setRange(-1.0, 1e6)
        self.th_min_spin.setDecimals(6)
        self.th_min_spin.setValue(-1.0)
        th_range_min_row.addWidget(self.th_min_spin)

        th_range_max_row = QHBoxLayout()
        th_range_max_row.addWidget(QLabel("Thickness max"))
        self.th_max_spin = QDoubleSpinBox()
        self.th_max_spin.setRange(-1.0, 1e6)
        self.th_max_spin.setDecimals(6)
        self.th_max_spin.setValue(-1.0)
        th_range_max_row.addWidget(self.th_max_spin)

        th_layout.addLayout(th_target_row)
        th_layout.addLayout(th_tol_row)
        th_layout.addLayout(th_range_min_row)
        th_layout.addLayout(th_range_max_row)
        th_layout.insertWidget(0, self.th_enabled_checkbox)

        # Mode defaults
        self.rs_target_radio.setChecked(False)
        self.rs_range_radio.setChecked(True)
        self.th_target_radio.setChecked(False)
        self.th_range_radio.setChecked(True)

        # Apply button
        apply_button = QPushButton("Apply spec to current material")
        apply_button.clicked.connect(self._apply_clicked)

        # Wire UI enabling/disabling.
        self._update_enabled_states()
        self.rs_target_radio.toggled.connect(self._update_enabled_states)
        self.rs_range_radio.toggled.connect(self._update_enabled_states)
        self.th_target_radio.toggled.connect(self._update_enabled_states)
        self.th_range_radio.toggled.connect(self._update_enabled_states)
        self.rs_enabled_checkbox.toggled.connect(self._update_enabled_states)
        self.th_enabled_checkbox.toggled.connect(self._update_enabled_states)
        self.rsu_enabled_checkbox.toggled.connect(self._update_rsu_enabled_state)
        self.max_lifetime_checkbox.toggled.connect(self._update_max_lifetime_state)

        rs_group_layout = rs_layout
        rs_group.setLayout(rs_group_layout)
        th_group.setLayout(th_layout)
        rsu_group.setLayout(rsu_layout)

        layout.addWidget(rs_group)
        layout.addWidget(rsu_group)
        layout.addWidget(th_group)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def _update_max_lifetime_state(self) -> None:
        self.max_lifetime_spin.setEnabled(self.max_lifetime_checkbox.isChecked())

    def _update_rsu_enabled_state(self) -> None:
        enabled = self.rsu_enabled_checkbox.isChecked()
        self.rsu_max_spin.setEnabled(enabled)

    def _update_enabled_states(self) -> None:
        rs_target_mode = self.rs_target_radio.isChecked()
        th_target_mode = self.th_target_radio.isChecked()

        rs_enabled = self.rs_enabled_checkbox.isChecked()
        th_enabled = self.th_enabled_checkbox.isChecked()

        # RS inputs
        self.rs_target_radio.setEnabled(rs_enabled)
        self.rs_range_radio.setEnabled(rs_enabled)
        self.rs_target_spin.setEnabled(rs_enabled and rs_target_mode)
        self.rs_tol_spin.setEnabled(rs_enabled and rs_target_mode)
        self.rs_min_spin.setEnabled(rs_enabled and (not rs_target_mode))
        self.rs_max_spin.setEnabled(rs_enabled and (not rs_target_mode))

        self.th_target_radio.setEnabled(th_enabled)
        self.th_range_radio.setEnabled(th_enabled)
        # Thickness inputs
        self.th_target_spin.setEnabled(th_enabled and th_target_mode)
        self.th_tol_spin.setEnabled(th_enabled and th_target_mode)
        self.th_min_spin.setEnabled(th_enabled and (not th_target_mode))
        self.th_max_spin.setEnabled(th_enabled and (not th_target_mode))

    def _maybe_unset(self, value: float) -> Optional[float]:
        # Convention in UI: -1 means "unset".
        if value < 0:
            return None
        return float(value)

    def _apply_clicked(self) -> None:
        rs_target_mode = self.rs_target_radio.isChecked()
        th_target_mode = self.th_target_radio.isChecked()
        rs_enabled = self.rs_enabled_checkbox.isChecked()
        th_enabled = self.th_enabled_checkbox.isChecked()

        rs_target = None
        rs_tol = None
        rs_min = None
        rs_max = None
        if rs_target_mode:
            rs_target = self._maybe_unset(self.rs_target_spin.value())
            rs_tol = self._maybe_unset(self.rs_tol_spin.value())
        else:
            rs_min = self._maybe_unset(self.rs_min_spin.value())
            rs_max = self._maybe_unset(self.rs_max_spin.value())

        thickness_target = None
        thickness_tol = None
        thickness_min = None
        thickness_max = None
        if th_target_mode:
            thickness_target = self._maybe_unset(self.th_target_spin.value())
            thickness_tol = self._maybe_unset(self.th_tol_spin.value())
        else:
            thickness_min = self._maybe_unset(self.th_min_spin.value())
            thickness_max = self._maybe_unset(self.th_max_spin.value())

        rsu_enabled = self.rsu_enabled_checkbox.isChecked()
        rsu_max = float(self.rsu_max_spin.value()) if rsu_enabled else 0.0

        spec_config = SpecConfig(
            rs_target=rs_target,
            rs_min=rs_min,
            rs_max=rs_max,
            rs_tol=rs_tol,
            thickness_target=thickness_target,
            thickness_min=thickness_min,
            thickness_max=thickness_max,
            thickness_tol=thickness_tol,
            rsu_max=rsu_max,
            use_rs_target_mode=rs_target_mode,
            use_thickness_target_mode=th_target_mode,
            use_rs_spec=rs_enabled,
            use_thickness_spec=th_enabled,
            use_rsu_spec=rsu_enabled,
        )

        self.specApplied.emit(
            {
                "spec_config": spec_config,
                "save_per_target": bool(self.save_per_target_checkbox.isChecked()),
                "max_lifetime_enabled": bool(self.max_lifetime_checkbox.isChecked()),
                "max_lifetime_value": float(self.max_lifetime_spin.value()),
            }
        )

    def set_spec_config(self, spec_config: SpecConfig) -> None:
        """Update the panel from a SpecConfig object."""
        self.rs_enabled_checkbox.setChecked(spec_config.use_rs_spec)
        self.th_enabled_checkbox.setChecked(spec_config.use_thickness_spec)

        self.rs_target_radio.setChecked(spec_config.use_rs_target_mode)
        self.rs_range_radio.setChecked(not spec_config.use_rs_target_mode)
        self.th_target_radio.setChecked(spec_config.use_thickness_target_mode)
        self.th_range_radio.setChecked(not spec_config.use_thickness_target_mode)

        # RS
        self.rs_target_spin.setValue(float(spec_config.rs_target) if spec_config.rs_target is not None else -1.0)
        self.rs_tol_spin.setValue(float(spec_config.rs_tol) if spec_config.rs_tol is not None else -1.0)
        self.rs_min_spin.setValue(float(spec_config.rs_min) if spec_config.rs_min is not None else -1.0)
        self.rs_max_spin.setValue(float(spec_config.rs_max) if spec_config.rs_max is not None else -1.0)

        # Thickness
        self.th_target_spin.setValue(float(spec_config.thickness_target) if spec_config.thickness_target is not None else -1.0)
        self.th_tol_spin.setValue(float(spec_config.thickness_tol) if spec_config.thickness_tol is not None else -1.0)
        self.th_min_spin.setValue(float(spec_config.thickness_min) if spec_config.thickness_min is not None else -1.0)
        self.th_max_spin.setValue(float(spec_config.thickness_max) if spec_config.thickness_max is not None else -1.0)

        # RSU
        rsu_enabled = bool(spec_config.use_rsu_spec)
        self.rsu_enabled_checkbox.setChecked(rsu_enabled)
        self.rsu_max_spin.setValue(float(spec_config.rsu_max) if rsu_enabled else 0.0)

        self._update_enabled_states()
        self._update_rsu_enabled_state()
        self._update_max_lifetime_state()

