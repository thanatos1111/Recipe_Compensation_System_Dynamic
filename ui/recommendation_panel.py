"""
Recommendation panel (Milestone 5).

Generates discrete recipe candidates around the latest actual recipe,
predicts outputs using trained models + instance correction, ranks candidates,
and shows the best next recipe plus safe-band interval.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RecommendationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._material_dataset = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Recommendation panel (Milestone 5)"))

        # Lifetime input: start lifetime for the requested deposition state.
        lifetime_row = QHBoxLayout()
        lifetime_row.addWidget(QLabel("Current lifetime (start):"))
        self.lifetime_spin = QDoubleSpinBox()
        self.lifetime_spin.setRange(0.0, 1e9)
        self.lifetime_spin.setDecimals(6)
        self.lifetime_spin.setValue(0.0)
        lifetime_row.addWidget(self.lifetime_spin)
        layout.addLayout(lifetime_row)

        self.radius_steps_spin = QDoubleSpinBox()
        self.radius_steps_spin.setRange(0, 10)
        self.radius_steps_spin.setDecimals(0)
        self.radius_steps_spin.setValue(1)
        layout.addWidget(QLabel("Neighborhood radius (steps):"))
        layout.addWidget(self.radius_steps_spin)

        self.max_candidates_spin = QDoubleSpinBox()
        self.max_candidates_spin.setRange(1, 5000)
        self.max_candidates_spin.setDecimals(0)
        self.max_candidates_spin.setValue(200)
        layout.addWidget(QLabel("Max candidates (cap):"))
        layout.addWidget(self.max_candidates_spin)

        # Safe band param picker.
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Safe band param:"))
        self.band_param_combo = QComboBox()
        for p in [
            "incident_angle",
            "linear_offset",
            "rotations",
            "ar_flow",
            "o2_flow",
            "rpm",
            "power",
        ]:
            self.band_param_combo.addItem(p)
        band_row.addWidget(self.band_param_combo)
        layout.addLayout(band_row)

        self.generate_button = QPushButton("Generate recommendation")
        self.generate_button.clicked.connect(self._generate_clicked)
        layout.addWidget(self.generate_button)

        layout.addWidget(QLabel("Ranked candidates (top):"))
        self.candidates_table = QTableWidget()
        layout.addWidget(self.candidates_table)

        layout.addWidget(QLabel("Recommended next recipe:"))
        self.recommended_label = QLabel("")
        layout.addWidget(self.recommended_label)

        layout.addWidget(QLabel("Safe band (predicted feasible interval):"))
        self.safe_band_label = QLabel("")
        layout.addWidget(self.safe_band_label)

        layout.setStretchFactor(self.candidates_table, 1)

        self.setLayout(layout)

    def set_context(self, material_dataset: Any, *, active_target_id: str, config: dict[str, Any]) -> None:
        self._material_dataset = material_dataset
        self._active_target_id = active_target_id
        self._config = config

        # Initialize lifetime to the latest observed starting lifetime.
        try:
            inst = material_dataset.target_instances.get(active_target_id)
            if inst is not None and inst.records is not None and not inst.records.empty and "lifetime" in inst.records.columns:
                last = inst.records.sort_values("lifetime").iloc[-1]
                if "lifetime_end" in inst.records.columns and pd.notna(last.get("lifetime_end")):
                    self.lifetime_spin.setValue(float(last["lifetime_end"]))
                else:
                    self.lifetime_spin.setValue(float(last["lifetime"]))
        except Exception:
            pass

    def _generate_clicked(self) -> None:
        if self._material_dataset is None or not self._active_target_id:
            return

        artifacts = self._material_dataset.material_model_artifacts
        if artifacts is None or artifacts.rs_model is None or artifacts.thickness_model is None or artifacts.rsu_model is None:
            self.recommended_label.setText("Train models in Model tab first.")
            return

        inst = self._material_dataset.target_instances.get(self._active_target_id)
        if inst is None or inst.records is None or inst.records.empty:
            self.recommended_label.setText("No records for active target instance.")
            return

        ref_row = inst.records.sort_values("lifetime").iloc[-1]
        reference_recipe = {
            "incident_angle": float(ref_row.get("incident_angle", 0.0)),
            "linear_offset": float(ref_row.get("linear_offset", 0.0)),
            "rotations": float(ref_row.get("rotations", 0.0)),
            "ar_flow": float(ref_row.get("ar_flow", 0.0)),
            "o2_flow": float(ref_row.get("o2_flow", 0.0)),
            "total_flow": float(ref_row.get("ar_flow", 0.0)) + float(ref_row.get("o2_flow", 0.0)),
        }

        rpm = float(ref_row.get("rpm", 0.0)) if "rpm" in ref_row else None
        power = float(ref_row.get("power", 0.0)) if "power" in ref_row else None

        lifetime = float(self.lifetime_spin.value())
        spec_config = self._material_dataset.spec_config

        # Imports here to keep UI module import-time light.
        from core.candidate_generator import generate_candidates
        from core.optimizer import rank_candidates, select_best_candidate
        from core.safe_band import estimate_parameter_band

        parameter_config = self._config.get("parameter_constraints", {})
        min_steps = self._config.get("minimum_steps", {}) or {}

        def default_min_step(param_name: str) -> float:
            return 1.0 if param_name == "rotations" else 0.01

        # Ensure candidate generation + quantization always respect minimum steps.
        effective_parameter_config: dict[str, Any] = {}
        for pname, pcfg in (parameter_config or {}).items():
            if not isinstance(pcfg, dict):
                effective_parameter_config[pname] = pcfg
                continue
            eff = dict(pcfg)
            current_step = float(eff.get("step", 0.0) or 0.0)
            ms = float(min_steps.get(pname, default_min_step(pname)))
            if current_step <= 0:
                eff["step"] = ms
            else:
                eff["step"] = max(current_step, ms)
            effective_parameter_config[pname] = eff
        neighborhood_config = {
            "radius_steps": int(self.radius_steps_spin.value()),
            "max_candidates": int(self.max_candidates_spin.value()),
            "min_steps": min_steps,
        }

        candidates = generate_candidates(reference_recipe, effective_parameter_config, neighborhood_config)
        if candidates is None or candidates.empty:
            self.recommended_label.setText("No candidates generated (check neighborhood).")
            return

        model_bundle = {
            "rs_model": artifacts.rs_model,
            "thickness_model": artifacts.thickness_model,
            "rsu_model": artifacts.rsu_model,
            "feature_config": self._config.get("feature_config", {}),
        }
        # Pass instance-correction artifacts to optimizer.
        model_bundle["instance_correction"] = inst.instance_correction_artifacts

        context = {
            "spec_config": spec_config,
            "weights": self._config.get("optimizer_weights", {}),
            "reference_recipe": reference_recipe,
            "active_target_id": self._active_target_id,
            "lifetime": lifetime,
            "rpm": rpm,
            "power": power,
            "feature_config": self._config.get("feature_config", {}),
            "instance_correction": inst.instance_correction_artifacts,
            "parameter_config": effective_parameter_config,
        }

        ranked = rank_candidates(candidates, model_bundle, context)
        if ranked is None or ranked.empty:
            self.recommended_label.setText("No valid candidates after constraints/scoring.")
            return

        best = select_best_candidate(ranked)

        top_n = min(25, len(ranked))
        show = ranked.head(top_n)
        show_cols = [
            c
            for c in [
                "score",
                "incident_angle",
                "linear_offset",
                "rotations",
                "ar_flow",
                "o2_flow",
                "rs_pred",
                "thickness_pred",
                "rsu_pred",
            ]
            if c in show.columns
        ]

        self.candidates_table.clear()
        self.candidates_table.setRowCount(len(show))
        self.candidates_table.setColumnCount(len(show_cols))
        for ci, col_name in enumerate(show_cols):
            self.candidates_table.setHorizontalHeaderItem(ci, QTableWidgetItem(str(col_name)))
        for ri in range(len(show)):
            for ci, col_name in enumerate(show_cols):
                val = show.iloc[ri][col_name]
                text = "" if pd.isna(val) else str(val)
                self.candidates_table.setItem(ri, ci, QTableWidgetItem(text))

        self.recommended_label.setText(
            "incident_angle={ia}, linear_offset={lo}, rotations={rot}, ar_flow={ar}, o2_flow={o2}".format(
                ia=best.get("incident_angle", None),
                lo=best.get("linear_offset", None),
                rot=best.get("rotations", None),
                ar=best.get("ar_flow", None),
                o2=best.get("o2_flow", None),
            )
        )

        # Safe band estimation for selected parameter around reference recipe.
        band_param = self.band_param_combo.currentText()
        band_step = 0.0
        if band_param in effective_parameter_config and isinstance(effective_parameter_config[band_param], dict):
            band_step = float(effective_parameter_config[band_param].get("step", 0.0) or 0.0)
        if band_step <= 0:
            ref_v = reference_recipe.get(band_param, 0.0) or 0.0
            band_step = abs(float(ref_v)) * 0.05
            if band_step <= 0:
                band_step = 1.0

        center_recipe = dict(reference_recipe)
        center_recipe.update(
            {
                "target_id": self._active_target_id,
                "lifetime": lifetime,
                "rpm": rpm,
                "power": power,
            }
        )
        safe_band = estimate_parameter_band(
            center_recipe,
            model_bundle,
            parameter_name=band_param,
            spec_config=spec_config,
            step=band_step,
            max_steps=3,
        )
        lower = safe_band.get("lower")
        upper = safe_band.get("upper")
        if lower is None or upper is None:
            self.safe_band_label.setText("No feasible safe band found (predicted).")
        else:
            self.safe_band_label.setText(f"[{lower:.6g}, {upper:.6g}]")

