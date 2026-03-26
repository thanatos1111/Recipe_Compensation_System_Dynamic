"""
Model training/metrics panel (placeholder).
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from core.instance_correction import fit_instance_bias, summarize_recent_residuals
from core.response_models import train_material_models
from core.schemas import MaterialDataset


class ModelPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._material_dataset: Optional[MaterialDataset] = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model panel (Milestone 4)"))

        self.train_button = QPushButton("Train material models (RS/Thickness/RSU)")
        self.train_button.clicked.connect(self._train_clicked)
        layout.addWidget(self.train_button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        layout.setStretchFactor(self.output, 1)

        self.setLayout(layout)

    def set_context(self, material_dataset: MaterialDataset, *, active_target_id: str, config: dict[str, Any]) -> None:
        self._material_dataset = material_dataset
        self._active_target_id = active_target_id
        self._config = config

    def _train_clicked(self) -> None:
        if self._material_dataset is None:
            self.output.setPlainText("No material selected.")
            return

        df = self._material_dataset.all_records
        if df is None or df.empty:
            self.output.setPlainText("No rows available for training.")
            return

        # Train per-material models using all rows (including out-of-spec).
        artifacts = train_material_models(df, config=self._config)
        self._material_dataset.material_model_artifacts = artifacts

        # Fit active-instance correction if possible.
        correction_text = ""
        if self._active_target_id and self._active_target_id in self._material_dataset.target_instances:
            inst_df = self._material_dataset.target_instances[self._active_target_id].records
            model_bundle = {
                "rs_model": artifacts.rs_model,
                "thickness_model": artifacts.thickness_model,
                "rsu_model": artifacts.rsu_model,
                "feature_config": self._config.get("feature_config", {}),
            }
            correction = fit_instance_bias(model_bundle, inst_df)
            self._material_dataset.target_instances[self._active_target_id].instance_correction_artifacts = correction
            drift = summarize_recent_residuals(inst_df, model_bundle)
            correction_text = f"\n\nActive instance correction:\n- confidence: {correction.instance_confidence}\n- bias_terms: {correction.bias_terms}\n- recent_residuals: {drift}"

        self.output.setPlainText(
            "Training summary:\n"
            f"- material: {self._material_dataset.material_name}\n"
            f"- rows: {artifacts.train_summary.get('row_count')}\n"
            f"- target_instances: {artifacts.train_summary.get('target_instance_count')}\n"
            "\nMetrics (time-aware holdout):\n"
            f"- rs_mae: {artifacts.metrics.get('rs_mae')}\n"
            f"- thickness_mae: {artifacts.metrics.get('thickness_mae')}\n"
            f"- rsu_mae: {artifacts.metrics.get('rsu_mae')}\n"
            "\nConfidence:\n"
            f"- level: {artifacts.confidence_summary.get('level')}\n"
            f"- notes: {artifacts.confidence_summary.get('notes')}\n"
            f"{correction_text}"
        )

