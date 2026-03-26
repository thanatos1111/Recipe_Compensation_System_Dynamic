"""
Main application window.

Milestone 3 adds:
- spec configuration + in-spec/out-of-spec labeling
- trend plots with in-spec highlighting
- baseline reference tables by lifetime bin
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from core.material_manager import MaterialManager
from core.labeling import compute_derived_features
from core.schemas import SpecConfig

from ui.material_selector import MaterialSelectorPanel
from ui.target_instance_selector import TargetInstanceSelectorPanel
from ui.raw_table_panel import RawTablePanel
from ui.spec_config_panel import SpecConfigPanel
from ui.trend_panel import TrendPanel
from ui.model_panel import ModelPanel
from ui.recommendation_panel import RecommendationPanel
from ui.update_panel import UpdatePanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Recipe Compensation System (Milestone 3)")

        self.material_manager = MaterialManager()
        self._current_material: Optional[str] = None
        self._current_target_id: Optional[str] = None
        self._spec_config: SpecConfig = SpecConfig()

        self.config = self._load_default_config()
        self.column_map = self.config.get("column_mapping", {})
        self._spec_config = self._spec_config_from_config(self.config)

        central = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Milestone 3: Excel import + labeling + baseline/plots."))

        self.material_selector = MaterialSelectorPanel()
        self.target_instance_selector = TargetInstanceSelectorPanel()
        self.raw_table_panel = RawTablePanel()

        self.spec_panel = SpecConfigPanel()
        self.spec_panel.set_spec_config(self._spec_config)
        self.spec_panel.specApplied.connect(self._on_spec_applied)

        self.material_selector.set_on_open_workbook(self._open_workbook)
        self.material_selector.materialSelected.connect(self._on_material_selected)
        self.target_instance_selector.targetSelected.connect(self._on_target_selected)

        self.trend_panel = TrendPanel()

        # Tabbed layout to avoid vertical "squeezy" stacking (matches spec panels A–H).
        self.tabs = QTabWidget()
        self.tabs.setMovable(True)

        # Operational order A -> H (matches spec workflow).
        self.tabs.addTab(self.material_selector, "A. Workbook/material")
        self.tabs.addTab(self.target_instance_selector, "B. Target instance")
        self.tabs.addTab(self.raw_table_panel, "C. Raw data")
        self.tabs.addTab(self.spec_panel, "D. Spec config")

        self.tab_trends_index = self.tabs.addTab(self.trend_panel, "E. Trends")
        self.tabs.addTab(ModelPanel(), "F. Model")
        self.tabs.addTab(RecommendationPanel(), "G. Recommendation")
        self.tabs.addTab(UpdatePanel(), "H. Update")

        # Ensure plotting refresh when switching back to Trends.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def _on_tab_changed(self, _index: int) -> None:
        if _index != self.tab_trends_index:
            return
        if not self._current_material or not self._current_target_id:
            return
        dataset = self.material_manager.get_material(self._current_material)
        if self._current_target_id in dataset.target_instances:
            self.trend_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)

    def _load_default_config(self) -> dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "default_config.json"
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _spec_config_from_config(self, cfg: dict[str, Any]) -> SpecConfig:
        spec = cfg.get("spec_settings", {})
        rsu_max = float(spec.get("rsu_max", 0.0) or 0.0)
        return SpecConfig(
            rs_target=spec.get("rs_target", None),
            rs_min=spec.get("rs_min", None),
            rs_max=spec.get("rs_max", None),
            rs_tol=spec.get("rs_tol", None),
            thickness_target=spec.get("thickness_target", None),
            thickness_min=spec.get("thickness_min", None),
            thickness_max=spec.get("thickness_max", None),
            thickness_tol=spec.get("thickness_tol", None),
            rsu_max=rsu_max,
            use_rs_target_mode=bool(spec.get("use_rs_target_mode", False)),
            use_thickness_target_mode=bool(spec.get("use_thickness_target_mode", False)),
            # Default enable flags based on numeric thresholds.
            use_rs_spec=True,
            use_thickness_spec=True,
            use_rsu_spec=rsu_max > 0,
        )

    def _open_workbook(self, workbook_path: str) -> None:
        self.material_manager.clear()
        materials = self.material_manager.load_workbook(
            workbook_path,
            column_map=self.column_map,
            config=self.config,
        )

        self.material_selector.set_materials(materials)
        if materials:
            self._on_material_selected(materials[0])

    def _on_material_selected(self, material_name: str) -> None:
        self._current_material = material_name
        dataset = self.material_manager.get_material(material_name)
        self._apply_spec_and_derived_to_material(dataset)

        target_ids = self.material_manager.get_target_instance_ids(material_name)
        self.target_instance_selector.set_targets(target_ids)
        self.raw_table_panel.set_target_ids(target_ids, active_target_id=target_ids[0] if target_ids else None)
        self.raw_table_panel.set_material_dataframe(dataset.all_records)

        if target_ids:
            self._on_target_selected(target_ids[0])

    def _on_target_selected(self, target_id: str) -> None:
        if not self._current_material:
            return
        self._current_target_id = target_id
        dataset = self.material_manager.get_material(self._current_material)
        instance = dataset.target_instances.get(target_id)
        if instance is None:
            self.raw_table_panel.set_dataframe(None)  # type: ignore[arg-type]
            return
        self.raw_table_panel.set_active_target_id(target_id)
        # Ensure the trend panel uses labeled records.
        self.trend_panel.set_context(dataset, active_target_id=target_id, config=self.config)

    def _on_spec_applied(self, spec_config: SpecConfig) -> None:
        self._spec_config = spec_config
        if not self._current_material:
            return

        dataset = self.material_manager.get_material(self._current_material)
        self._apply_spec_and_derived_to_material(dataset)

        # Refresh table + plots with the updated labels.
        self.raw_table_panel.set_material_dataframe(dataset.all_records)

        active_target_id = self._current_target_id
        if active_target_id and active_target_id in dataset.target_instances:
            self.trend_panel.set_context(dataset, active_target_id=active_target_id, config=self.config)

    def _apply_spec_and_derived_to_material(self, dataset: Any) -> None:
        """
        Recompute derived columns + `in_spec` for the given material dataset.
        """
        labeled = compute_derived_features(dataset.all_records, self._spec_config)
        dataset.all_records = labeled
        dataset.spec_config = self._spec_config

        # Update each target instance's record view from the material-level table.
        if "target_id" in labeled.columns:
            for target_id, inst in dataset.target_instances.items():
                if "target_id" not in labeled.columns:
                    inst.records = inst.records
                    continue
                subset = labeled[labeled["target_id"] == target_id].copy()
                if "lifetime" in subset.columns:
                    subset = subset.sort_values("lifetime").reset_index(drop=True)
                inst.records = subset

