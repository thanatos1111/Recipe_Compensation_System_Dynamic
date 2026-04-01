"""
Main application window.

Milestone 3 adds:
- spec configuration + in-spec/out-of-spec labeling
- trend plots with in-spec highlighting
- baseline reference tables by lifetime bin
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PySide6.QtWidgets import QLabel, QMainWindow, QMenu, QMenuBar, QTabWidget, QVBoxLayout, QWidget

from core.config_store import load_effective_config, load_user_config, save_user_config
from core.material_manager import MaterialManager
from core.parameter_registry import apply_parameter_registry_to_config
from core.scoped_settings import (
    apply_scoped_parameter_constraints_to_config,
    load_effective_scoped_settings,
    resolve_effective_max_lifetime,
    resolve_effective_spec_dict,
)
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
from ui.app_settings_dialog import AppSettingsDialog


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Recipe Compensation System (Milestone 3)")

        self.material_manager = MaterialManager()
        self._current_material: Optional[str] = None
        self._current_target_id: Optional[str] = None
        self._spec_config: SpecConfig = SpecConfig()
        self._max_lifetime: Optional[float] = None

        self.project_root = Path(__file__).resolve().parents[1]
        self.user_config = load_user_config(self.project_root)

        self.config = load_effective_config(self.project_root)
        apply_parameter_registry_to_config(self.project_root, self.config)
        self._scoped = load_effective_scoped_settings(self.project_root, self.user_config)
        self.column_map = self.config.get("column_mapping", {})
        self._spec_config = self._spec_config_from_config(self.config.get("spec_settings", {}))

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
        self.model_panel = ModelPanel()
        self.recommendation_panel = RecommendationPanel()
        self.update_panel = UpdatePanel()
        self.update_panel.refreshRequested.connect(self._refresh_all_panels_from_store)

        self._setup_menu()

        # Tabbed layout to avoid vertical "squeezy" stacking (matches spec panels A–H).
        self.tabs = QTabWidget()
        self.tabs.setMovable(True)

        # Operational order A -> H (matches spec workflow).
        self.tabs.addTab(self.material_selector, "A. Workbook/material")
        self.tabs.addTab(self.target_instance_selector, "B. Target instance")
        self.tabs.addTab(self.raw_table_panel, "C. Raw data")
        self.tabs.addTab(self.spec_panel, "D. Spec config")

        self.tab_trends_index = self.tabs.addTab(self.trend_panel, "E. Trends")
        self.tab_model_index = self.tabs.addTab(self.model_panel, "F. Model")
        self.tabs.addTab(self.recommendation_panel, "G. Recommendation")
        self.tabs.addTab(self.update_panel, "H. Update")

        # Ensure plotting refresh when switching back to Trends.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # Persisted last workbook auto-load + picker default directory.
        last_path = self.user_config.get("last_workbook_path")
        if isinstance(last_path, str) and last_path.strip():
            p = Path(last_path)
            if p.exists():
                self.material_selector.set_initial_directory(str(p.parent))
                # Auto-load to skip manual step A.
                self._open_workbook(str(p))

    def _setup_menu(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        settings_menu = QMenu("Settings", self)
        menubar.addMenu(settings_menu)

        action_settings = settings_menu.addAction("Settings…")
        action_settings.triggered.connect(self._open_app_settings)

    def _open_app_settings(self) -> None:
        """Tabbed editor for parameter registry, aliases, minimum steps, and related options."""

        def on_apply(payload: dict[str, Any]) -> None:
            self.user_config["minimum_steps"] = payload.get("minimum_steps", {})
            self.user_config["model_validation_defaults"] = payload.get("model_validation_defaults", {})
            if "scoped_settings" in payload:
                self.user_config["scoped_settings"] = payload["scoped_settings"]
            save_user_config(self.project_root, self.user_config)
            self._sync_config_for_active_selection()

            if self._current_material and self._current_target_id:
                dataset = self.material_manager.get_material(self._current_material)
                self._spec_config = self._get_effective_spec_for_selection(
                    self._current_material, self._current_target_id
                )
                self._max_lifetime = self._get_effective_max_lifetime(
                    self._current_material, self._current_target_id
                )
                self.spec_panel.set_spec_config(self._spec_config)
                self._apply_spec_and_derived_to_material(dataset)
                self.raw_table_panel.set_material_dataframe(dataset.all_records)
                self.trend_panel.set_context(
                    dataset,
                    active_target_id=self._current_target_id,
                    config=self.config,
                )
                self.model_panel.set_context(
                    dataset,
                    active_target_id=self._current_target_id,
                    config=self.config,
                )
                self.recommendation_panel.set_context(
                    dataset,
                    active_target_id=self._current_target_id,
                    config=self.config,
                )
                self.update_panel.set_context(
                    dataset,
                    active_target_id=self._current_target_id,
                    config=self.config,
                )

        dlg = AppSettingsDialog(
            project_root=self.project_root,
            initial_user_config=self.user_config,
            on_apply=on_apply,
            parent=self,
        )
        dlg.exec()

    def _sync_config_for_active_selection(self) -> None:
        """Reload effective config, apply registry, then scoped parameter overrides for active target."""
        self.config = load_effective_config(self.project_root)
        apply_parameter_registry_to_config(self.project_root, self.config)
        self._scoped = load_effective_scoped_settings(self.project_root, self.user_config)
        if self._current_material and self._current_target_id:
            apply_scoped_parameter_constraints_to_config(
                self.project_root,
                self.config,
                material_name=self._current_material,
                target_id=self._current_target_id,
                scoped=self._scoped,
            )

    def _on_tab_changed(self, _index: int) -> None:
        if _index != self.tab_trends_index:
            return
        if not self._current_material or not self._current_target_id:
            return
        dataset = self.material_manager.get_material(self._current_material)
        if self._current_target_id in dataset.target_instances:
            self.trend_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)

    def _spec_config_from_config(self, spec: dict[str, Any]) -> SpecConfig:
        rsu_max = float(spec.get("rsu_max", 0.0) or 0.0)
        return SpecConfig(
            rs_target=spec.get("rs_target"),
            rs_min=spec.get("rs_min"),
            rs_max=spec.get("rs_max"),
            rs_tol=spec.get("rs_tol"),
            thickness_target=spec.get("thickness_target"),
            thickness_min=spec.get("thickness_min"),
            thickness_max=spec.get("thickness_max"),
            thickness_tol=spec.get("thickness_tol"),
            rsu_max=rsu_max,
            use_rs_target_mode=bool(spec.get("use_rs_target_mode", False)),
            use_thickness_target_mode=bool(spec.get("use_thickness_target_mode", False)),
            # Default enable flags based on numeric thresholds.
            use_rs_spec=bool(spec.get("use_rs_spec", True)),
            use_thickness_spec=bool(spec.get("use_thickness_spec", True)),
            use_rsu_spec=bool(spec.get("use_rsu_spec", rsu_max > 0)),
        )

    def _get_effective_spec_for_selection(self, material_name: str, target_id: Optional[str]) -> SpecConfig:
        merged = resolve_effective_spec_dict(
            self.config,
            self._scoped,
            material_name=material_name,
            target_id=target_id,
        )
        return self._spec_config_from_config(merged)

    def _get_effective_max_lifetime(self, material_name: str, target_id: Optional[str]) -> Optional[float]:
        return resolve_effective_max_lifetime(
            self.config,
            self._scoped,
            material_name=material_name,
            target_id=target_id,
        )

    def _open_workbook(self, workbook_path: str) -> None:
        # Persist last opened workbook path.
        if isinstance(workbook_path, str) and workbook_path:
            self.user_config["last_workbook_path"] = workbook_path
            save_user_config(self.project_root, self.user_config)
            p = Path(workbook_path)
            if p.exists():
                self.material_selector.set_initial_directory(str(p.parent))

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
        self._current_target_id = None
        dataset = self.material_manager.get_material(material_name)

        # Load effective spec/max_lifetime for this material (target-specific applied after target selection).
        self._spec_config = self._get_effective_spec_for_selection(material_name, None)
        self._max_lifetime = self._get_effective_max_lifetime(material_name, None)
        self.spec_panel.set_spec_config(self._spec_config)

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

        self._sync_config_for_active_selection()

        # Apply target-specific overrides for spec/max_lifetime (scoped + legacy).
        self._spec_config = self._get_effective_spec_for_selection(self._current_material, target_id)
        self._max_lifetime = self._get_effective_max_lifetime(self._current_material, target_id)
        self.spec_panel.set_spec_config(self._spec_config)

        self._apply_spec_and_derived_to_material(dataset)
        self.raw_table_panel.set_material_dataframe(dataset.all_records)

        self.raw_table_panel.set_active_target_id(target_id)
        # Ensure the trend panel uses labeled records.
        self.trend_panel.set_context(dataset, active_target_id=target_id, config=self.config)
        self.model_panel.set_context(dataset, active_target_id=target_id, config=self.config)
        self.recommendation_panel.set_context(dataset, active_target_id=target_id, config=self.config)
        self.update_panel.set_context(dataset, active_target_id=target_id, config=self.config)

    def _on_spec_applied(self, payload: Any) -> None:
        # Payload comes from SpecConfigPanel (spec + persistence options).
        if not isinstance(payload, dict) or "spec_config" not in payload:
            return
        spec_config: SpecConfig = payload["spec_config"]
        save_per_target = bool(payload.get("save_per_target", True))
        max_lt_enabled = bool(payload.get("max_lifetime_enabled", False))
        max_lt_value = float(payload.get("max_lifetime_value", 0.0))

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
            self.recommendation_panel.set_context(dataset, active_target_id=active_target_id, config=self.config)
            self.update_panel.set_context(dataset, active_target_id=active_target_id, config=self.config)

    def _refresh_all_panels_from_store(self) -> None:
        if not self._current_material or not self._current_target_id:
            return
        dataset = self.material_manager.get_material(self._current_material)

        # Ensure derived labels match the current spec.
        self._apply_spec_and_derived_to_material(dataset)

        self.raw_table_panel.set_material_dataframe(dataset.all_records)
        self.trend_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)
        self.model_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)
        self.recommendation_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)
        self.update_panel.set_context(dataset, active_target_id=self._current_target_id, config=self.config)

        # Reload effective config (so newly-saved overrides are reflected).
        self.user_config = load_user_config(self.project_root)
        self._sync_config_for_active_selection()

    def _persist_spec_and_lifetime_overrides(
        self,
        *,
        material_name: str,
        target_id: Optional[str],
        spec_config: SpecConfig,
        save_per_target: bool,
        max_lifetime: Optional[float],
    ) -> None:
        uc = dict(self.user_config) if isinstance(self.user_config, dict) else {}
        mo = uc.setdefault("material_overrides", {})
        mats = mo.setdefault("materials", {})
        mat_block = mats.setdefault(material_name, {})

        spec_dict = {
            "rs_target": spec_config.rs_target,
            "rs_min": spec_config.rs_min,
            "rs_max": spec_config.rs_max,
            "rs_tol": spec_config.rs_tol,
            "thickness_target": spec_config.thickness_target,
            "thickness_min": spec_config.thickness_min,
            "thickness_max": spec_config.thickness_max,
            "thickness_tol": spec_config.thickness_tol,
            "rsu_max": spec_config.rsu_max,
            "use_rs_target_mode": spec_config.use_rs_target_mode,
            "use_thickness_target_mode": spec_config.use_thickness_target_mode,
            "use_rs_spec": spec_config.use_rs_spec,
            "use_thickness_spec": spec_config.use_thickness_spec,
            "use_rsu_spec": spec_config.use_rsu_spec,
        }

        if save_per_target and target_id:
            targets = mat_block.setdefault("targets", {})
            tblock = targets.setdefault(target_id, {})
            tblock["spec_settings"] = spec_dict
            if max_lifetime is not None:
                tblock["max_lifetime"] = max_lifetime
            elif "max_lifetime" in tblock:
                # remove if disabled
                tblock.pop("max_lifetime", None)
        else:
            mat_block["spec_settings"] = spec_dict
            if max_lifetime is not None:
                mat_block["max_lifetime"] = max_lifetime
            elif "max_lifetime" in mat_block:
                mat_block.pop("max_lifetime", None)

        save_user_config(self.project_root, uc)

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

