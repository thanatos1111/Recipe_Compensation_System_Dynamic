"""
Update panel (Milestone 6).

Allows appending new deposition run rows, then refreshing derived labels,
retraining models, refitting instance correction, and showing recommendation
changes (before/after).
"""

from __future__ import annotations

import io
from typing import Any, Optional

import pandas as pd
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QDoubleSpinBox,
)

from core.recommendation_engine import recommend_next_recipe
from core.update_pipeline import append_runs_and_refresh
from core.schemas import SpecConfig


class UpdatePanel(QWidget):
    # Proper Qt signal declaration (class-level), so it has `.connect`.
    refreshRequested = Signal()

    def __init__(self) -> None:
        super().__init__()

        self._material_dataset: Any = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Append new runs + refresh (Milestone 6)"))

        # Lifetime for recommendation/recompute state.
        self.lifetime_start_spin = QDoubleSpinBox()
        self.lifetime_start_spin.setRange(0.0, 1e9)
        self.lifetime_start_spin.setDecimals(6)
        self.lifetime_start_spin.setValue(0.0)
        lifetime_row = QHBoxLayout()
        lifetime_row.addWidget(QLabel("Recommendation lifetime start:"))
        lifetime_row.addWidget(self.lifetime_start_spin)
        layout.addLayout(lifetime_row)

        self.input_new_only_checkbox = QCheckBox("If input has column `is_new`, append rows where is_new is true")
        self.input_new_only_checkbox.setChecked(True)
        layout.addWidget(self.input_new_only_checkbox)

        layout.addWidget(QLabel("Paste new rows as CSV (canonical columns recommended):"))
        self.csv_input = QTextEdit()
        self.csv_input.setPlaceholderText(
            "Example header: target_id,lifetime,incident_angle,linear_offset,rotations,rpm,power,ar_flow,o2_flow,thickness,rs,rsu,date,lot_id,wafer_id\n"
        )
        layout.addWidget(self.csv_input)

        self.append_button = QPushButton("Append rows + refresh modeling + show before/after")
        self.append_button.clicked.connect(self._append_clicked)
        layout.addWidget(self.append_button)

        layout.addWidget(QLabel("Before vs After (instance correction + best recipe):"))
        self.diff_output = QTextEdit()
        self.diff_output.setReadOnly(True)
        layout.addWidget(self.diff_output)

        self.setLayout(layout)

    def set_context(self, material_dataset: Any, *, active_target_id: str, config: dict[str, Any]) -> None:
        self._material_dataset = material_dataset
        self._active_target_id = active_target_id
        self._config = config

        # Initialize lifetime_start from last known starting lifetime or lifetime_end.
        try:
            inst = material_dataset.target_instances.get(active_target_id)
            if inst is not None and inst.records is not None and not inst.records.empty and "lifetime" in inst.records.columns:
                last = inst.records.sort_values("lifetime").iloc[-1]
                if "lifetime_end" in inst.records.columns and pd.notna(last.get("lifetime_end")):
                    self.lifetime_start_spin.setValue(float(last.get("lifetime_end")))
                else:
                    self.lifetime_start_spin.setValue(float(last.get("lifetime")))
        except Exception:
            pass

    def _parse_csv_rows(self, text: str) -> pd.DataFrame:
        csv_str = text.strip()
        if not csv_str:
            return pd.DataFrame()
        # Support both comma and tab separated by letting pandas sniff? Keep explicit CSV here.
        df = pd.read_csv(io.StringIO(csv_str))
        return df

    def _append_clicked(self) -> None:
        if self._material_dataset is None or not self._active_target_id:
            return

        ds = self._material_dataset
        active_target_id = self._active_target_id
        df_in = self._parse_csv_rows(self.csv_input.toPlainText())
        if df_in.empty:
            return

        # Optional: only append rows marked as new.
        if self.input_new_only_checkbox.isChecked() and "is_new" in df_in.columns:
            def to_bool(v: Any) -> bool:
                if isinstance(v, str):
                    return v.strip().lower() in {"true", "1", "yes", "y"}
                if isinstance(v, (int, float)):
                    return float(v) != 0.0
                return bool(v)

            mask = df_in["is_new"].apply(to_bool)
            df_new = df_in[mask].copy()
        else:
            df_new = df_in.copy()

        # Drop helper column if present.
        if "is_new" in df_new.columns:
            df_new = df_new.drop(columns=["is_new"])

        if df_new.empty:
            self.diff_output.setPlainText("No new rows selected by is_new filter.")
            return

        # Capture before recommendation and correction.
        lifetime_start = float(self.lifetime_start_spin.value())
        before_ranked, before_best = recommend_next_recipe(
            material_dataset=ds,
            active_target_id=active_target_id,
            lifetime_start=lifetime_start,
            config=self._config,
            neighborhood_config={"radius_steps": 1, "max_candidates": 200},
        )
        before_corr = None
        try:
            before_corr = ds.target_instances[active_target_id].instance_correction_artifacts
        except Exception:
            before_corr = None

        # Append + refresh (retrain models + refit correction).
        ds = append_runs_and_refresh(
            ds,
            active_target_id,
            df_new,
            train_config=self._config,
            spec_config=ds.spec_config,
        )

        # Trigger UI refresh.
        self.refreshRequested.emit()

        # After state.
        after_ranked, after_best = recommend_next_recipe(
            material_dataset=ds,
            active_target_id=active_target_id,
            lifetime_start=lifetime_start,
            config=self._config,
            neighborhood_config={"radius_steps": 1, "max_candidates": 200},
        )
        after_corr = None
        try:
            after_corr = ds.target_instances[active_target_id].instance_correction_artifacts
        except Exception:
            after_corr = None

        # Diff output.
        def fmt_candidate(best: dict[str, Any]) -> str:
            if not best:
                return "n/a"
            parts = []
            for k in ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"]:
                if k in best:
                    parts.append(f"{k}={best.get(k)}")
            outs = []
            for k in ["rs_pred", "thickness_pred", "rsu_pred"]:
                if k in best:
                    outs.append(f"{k}={best.get(k)}")
            return f"best: {', '.join(parts)}\noutputs: {', '.join(outs)}"

        def fmt_bias(corr: Any) -> str:
            if corr is None:
                return "n/a"
            bias_terms = getattr(corr, "bias_terms", None)
            conf = getattr(corr, "instance_confidence", None)
            return f"confidence={conf}, bias_terms={bias_terms}"

        diff_lines = [
            f"Appended rows: {len(df_new)}",
            "",
            "=== Before ===",
            fmt_bias(before_corr),
            fmt_candidate(before_best),
            "",
            "=== After ===",
            fmt_bias(after_corr),
            fmt_candidate(after_best),
        ]
        self.diff_output.setPlainText("\n".join(diff_lines))

