"""
Model training and evaluation panel.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.evaluation_mode import ScenarioConfig, evaluate_data_sufficiency_scenarios
from core.instance_correction import fit_instance_bias, summarize_recent_residuals
from core.response_models import train_material_models
from core.schemas import MaterialDataset

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    FigureCanvas = None  # type: ignore[assignment]

from matplotlib.figure import Figure


class _EvalWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        spec_config: Any,
        config: dict[str, Any],
        scenario_a: ScenarioConfig,
        scenario_b: ScenarioConfig,
    ) -> None:
        super().__init__()
        self._df = df
        self._spec_config = spec_config
        self._config = config
        self._scenario_a = scenario_a
        self._scenario_b = scenario_b

    def run(self) -> None:
        try:
            result = evaluate_data_sufficiency_scenarios(
                self._df,
                spec_config=self._spec_config,
                config=self._config,
                scenario_a=self._scenario_a,
                scenario_b=self._scenario_b,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._material_dataset: Optional[MaterialDataset] = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}
        self._eval_thread: Optional[QThread] = None
        self._eval_worker: Optional[_EvalWorker] = None
        self._ordered_target_ids: list[str] = []

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model and evaluation panel"))

        self.section_tabs = QTabWidget()
        layout.addWidget(self.section_tabs)

        # Section 1: training only.
        training_page = QWidget()
        training_layout = QVBoxLayout()
        train_context_row = QHBoxLayout()
        self.material_label = QLabel("-")
        train_context_row.addWidget(QLabel("Material:"))
        train_context_row.addWidget(self.material_label)
        train_context_row.addWidget(QLabel("Current target (from B tab):"))
        self.training_active_target_label = QLabel("-")
        train_context_row.addWidget(self.training_active_target_label)
        train_context_row.addStretch(1)
        training_layout.addLayout(train_context_row)

        self.train_button = QPushButton("Train material models (RS/Thickness/RSU)")
        self.train_button.clicked.connect(self._train_clicked)
        training_layout.addWidget(self.train_button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        training_layout.addWidget(self.output)
        training_layout.setStretchFactor(self.output, 1)
        training_page.setLayout(training_layout)
        self.section_tabs.addTab(training_page, "1) Training")

        # Section 2: scenario evaluation only.
        eval_page = QWidget()
        eval_layout = QVBoxLayout()

        eval_context_box = QGroupBox("Scenario evaluation setup")
        eval_context_form = QFormLayout()
        eval_context_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        eval_context_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        eval_context_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(QLabel("Active target:"))
        self.eval_active_target_combo = QComboBox()
        self.eval_active_target_combo.currentTextChanged.connect(self._on_active_target_changed)
        cutoff_row.addWidget(self.eval_active_target_combo)
        cutoff_row.addSpacing(16)
        cutoff_row.addWidget(QLabel("Cutoff lifetime on active target:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 1e9)
        self.cutoff_spin.setDecimals(6)
        self.cutoff_spin.setValue(0.0)
        cutoff_row.addWidget(self.cutoff_spin)
        cutoff_row.addStretch(1)
        eval_context_form.addRow(cutoff_row)

        scenarios_row = QHBoxLayout()

        scenario_a_box = QGroupBox("Scenario A")
        scenario_a_layout = QVBoxLayout()
        self.scenario_a_mode_combo = QComboBox()
        self.scenario_a_mode_combo.addItems(["All prior targets", "Selected targets"])
        self.scenario_a_mode_combo.currentTextChanged.connect(self._refresh_history_selection_enabled_state)
        scenario_a_layout.addWidget(QLabel("History source"))
        scenario_a_layout.addWidget(self.scenario_a_mode_combo)
        self.scenario_a_targets_list = QListWidget()
        self.scenario_a_targets_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.scenario_a_targets_list.setMinimumHeight(130)
        self.scenario_a_targets_list.setMinimumWidth(220)
        self.scenario_a_expand_btn = QToolButton()
        self.scenario_a_expand_btn.setText("Selected history targets")
        self.scenario_a_expand_btn.setCheckable(True)
        self.scenario_a_expand_btn.setChecked(False)
        self.scenario_a_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.scenario_a_expand_btn.toggled.connect(self._toggle_scenario_a_list)
        scenario_a_layout.addWidget(self.scenario_a_expand_btn)
        scenario_a_layout.addWidget(self.scenario_a_targets_list)
        self.scenario_a_targets_list.setVisible(False)
        scenario_a_box.setLayout(scenario_a_layout)
        scenarios_row.addWidget(scenario_a_box, 1)

        scenario_b_box = QGroupBox("Scenario B")
        scenario_b_layout = QVBoxLayout()
        self.scenario_b_mode_combo = QComboBox()
        self.scenario_b_mode_combo.addItems(["All prior targets", "Selected targets"])
        self.scenario_b_mode_combo.currentTextChanged.connect(self._refresh_history_selection_enabled_state)
        scenario_b_layout.addWidget(QLabel("History source"))
        scenario_b_layout.addWidget(self.scenario_b_mode_combo)
        self.scenario_b_targets_list = QListWidget()
        self.scenario_b_targets_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.scenario_b_targets_list.setMinimumHeight(130)
        self.scenario_b_targets_list.setMinimumWidth(220)
        self.scenario_b_expand_btn = QToolButton()
        self.scenario_b_expand_btn.setText("Selected history targets")
        self.scenario_b_expand_btn.setCheckable(True)
        self.scenario_b_expand_btn.setChecked(False)
        self.scenario_b_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.scenario_b_expand_btn.toggled.connect(self._toggle_scenario_b_list)
        scenario_b_layout.addWidget(self.scenario_b_expand_btn)
        scenario_b_layout.addWidget(self.scenario_b_targets_list)
        self.scenario_b_targets_list.setVisible(False)
        scenario_b_box.setLayout(scenario_b_layout)
        scenarios_row.addWidget(scenario_b_box, 1)

        eval_context_form.addRow(scenarios_row)

        action_row = QHBoxLayout()
        self.evaluate_button = QPushButton("Run scenario comparison")
        self.evaluate_button.clicked.connect(self._evaluate_clicked)
        self.eval_status_label = QLabel("")
        action_row.addWidget(self.evaluate_button)
        action_row.addWidget(self.eval_status_label)
        action_row.addStretch(1)
        eval_context_form.addRow(action_row)
        eval_context_box.setLayout(eval_context_form)
        eval_layout.addWidget(eval_context_box)

        self.main_splitter = QSplitter()
        self.main_splitter.setOrientation(Qt.Orientation.Vertical)
        eval_top = QSplitter()
        eval_top.setOrientation(Qt.Orientation.Horizontal)

        self.metrics_table = QTableWidget()
        self.metrics_table.setMinimumHeight(260)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        eval_top.addWidget(self.metrics_table)

        self.figure = Figure(figsize=(6, 4))
        self.ax = self.figure.add_subplot(111)
        if FigureCanvas is not None:
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(260)
            eval_top.addWidget(self.canvas)
        else:
            self.canvas = None
        self.main_splitter.addWidget(eval_top)

        self.interpretation = QTextEdit()
        self.interpretation.setReadOnly(True)
        self.main_splitter.addWidget(self.interpretation)
        self.main_splitter.setSizes([340, 180])
        eval_layout.addWidget(self.main_splitter)
        eval_page.setLayout(eval_layout)
        self.section_tabs.addTab(eval_page, "2) Scenario Evaluation")

        self.setLayout(layout)

    def set_context(self, material_dataset: MaterialDataset, *, active_target_id: str, config: dict[str, Any]) -> None:
        self._material_dataset = material_dataset
        self._active_target_id = active_target_id
        self._config = config
        self.material_label.setText(material_dataset.material_name)
        target_ids = self._ordered_targets(material_dataset)
        self._ordered_target_ids = target_ids
        self.training_active_target_label.setText(active_target_id)
        self._set_combo_items(self.eval_active_target_combo, target_ids, active_target_id)
        self._refresh_history_target_lists(active_target_id)
        self._init_cutoff(material_dataset, active_target_id)
        self._refresh_history_selection_enabled_state()

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

    def _set_combo_items(self, combo: QComboBox, items: list[str], selected: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        for it in items:
            combo.addItem(it)
        if selected in items:
            combo.setCurrentText(selected)
        combo.blockSignals(False)

    def _set_target_list_items(self, widget: QListWidget, target_ids: list[str]) -> None:
        widget.clear()
        for tid in target_ids:
            widget.addItem(QListWidgetItem(tid))

    def _ordered_targets(self, dataset: MaterialDataset) -> list[str]:
        rows: list[tuple[float, str]] = []
        for target_id, inst in dataset.target_instances.items():
            if inst.records is None or inst.records.empty or "lifetime" not in inst.records.columns:
                rows.append((float("inf"), target_id))
                continue
            lt = pd.to_numeric(inst.records["lifetime"], errors="coerce").dropna()
            min_lt = float(lt.min()) if not lt.empty else float("inf")
            rows.append((min_lt, target_id))
        rows.sort(key=lambda x: (x[0], x[1]))
        return [tid for _, tid in rows]

    def _prior_target_ids_for_active(self, active_target_id: str) -> list[str]:
        if active_target_id not in self._ordered_target_ids:
            return []
        idx = self._ordered_target_ids.index(active_target_id)
        return self._ordered_target_ids[:idx]

    def _refresh_history_selection_enabled_state(self) -> None:
        a_selected_mode = self.scenario_a_mode_combo.currentText() == "Selected targets"
        b_selected_mode = self.scenario_b_mode_combo.currentText() == "Selected targets"
        self.scenario_a_expand_btn.setEnabled(a_selected_mode)
        self.scenario_b_expand_btn.setEnabled(b_selected_mode)
        self.scenario_a_targets_list.setEnabled(a_selected_mode)
        self.scenario_b_targets_list.setEnabled(b_selected_mode)
        if not a_selected_mode:
            self.scenario_a_expand_btn.setChecked(False)
        if not b_selected_mode:
            self.scenario_b_expand_btn.setChecked(False)

    def _selected_targets_from_widget(self, widget: QListWidget) -> list[str]:
        return [item.text() for item in widget.selectedItems()]

    def _on_active_target_changed(self, target_id: str) -> None:
        if self._material_dataset is None:
            return
        self.training_active_target_label.setText(target_id)
        self._refresh_history_target_lists(target_id)
        self._init_cutoff(self._material_dataset, target_id)

    def _refresh_history_target_lists(self, active_target_id: str) -> None:
        prior_targets = self._prior_target_ids_for_active(active_target_id)
        self._set_target_list_items(self.scenario_a_targets_list, prior_targets)
        self._set_target_list_items(self.scenario_b_targets_list, prior_targets)

    def _toggle_scenario_a_list(self, expanded: bool) -> None:
        self.scenario_a_targets_list.setVisible(expanded and self.scenario_a_targets_list.isEnabled())
        self.scenario_a_expand_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _toggle_scenario_b_list(self, expanded: bool) -> None:
        self.scenario_b_targets_list.setVisible(expanded and self.scenario_b_targets_list.isEnabled())
        self.scenario_b_expand_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _init_cutoff(self, dataset: MaterialDataset, target_id: str) -> None:
        inst = dataset.target_instances.get(target_id)
        if inst is None or inst.records is None or inst.records.empty or "lifetime" not in inst.records.columns:
            return
        lifetimes = pd.to_numeric(inst.records["lifetime"], errors="coerce").dropna()
        if lifetimes.empty:
            return
        self.cutoff_spin.setMinimum(float(lifetimes.min()))
        self.cutoff_spin.setMaximum(float(lifetimes.max()))
        self.cutoff_spin.setValue(float(lifetimes.quantile(0.5)))

    def _evaluate_clicked(self) -> None:
        if self._material_dataset is None:
            self.interpretation.setPlainText("No material selected.")
            return
        df = self._material_dataset.all_records
        if df is None or df.empty:
            self.interpretation.setPlainText("No rows available.")
            return
        if "target_id" not in df.columns or "lifetime" not in df.columns:
            self.interpretation.setPlainText("Evaluation requires target_id and lifetime columns.")
            return

        active_target_id = self.eval_active_target_combo.currentText().strip()
        if not active_target_id:
            self.interpretation.setPlainText("Select an active target.")
            return
        cutoff = float(self.cutoff_spin.value())

        prior_targets = self._prior_target_ids_for_active(active_target_id)
        if self.scenario_a_mode_combo.currentText() == "All prior targets":
            scenario_a_history = tuple(prior_targets)
        else:
            scenario_a_history = tuple(self._selected_targets_from_widget(self.scenario_a_targets_list))

        if self.scenario_b_mode_combo.currentText() == "All prior targets":
            scenario_b_history = tuple(prior_targets)
        else:
            scenario_b_history = tuple(self._selected_targets_from_widget(self.scenario_b_targets_list))

        scenario_a_history = tuple(t for t in scenario_a_history if t != active_target_id)
        scenario_b_history = tuple(t for t in scenario_b_history if t != active_target_id)

        scenario_a = ScenarioConfig(
            name="Scenario A",
            active_target_id=active_target_id,
            history_target_ids=scenario_a_history,
            active_cutoff_lifetime=cutoff,
        )
        scenario_b = ScenarioConfig(
            name="Scenario B",
            active_target_id=active_target_id,
            history_target_ids=scenario_b_history,
            active_cutoff_lifetime=cutoff,
        )

        self._start_eval_worker(df, scenario_a, scenario_b)

    def _start_eval_worker(self, df: pd.DataFrame, scenario_a: ScenarioConfig, scenario_b: ScenarioConfig) -> None:
        if self._eval_thread is not None and self._eval_thread.isRunning():
            self.eval_status_label.setText("Evaluation already running...")
            return

        self.evaluate_button.setEnabled(False)
        self.eval_status_label.setText("Running scenario comparison...")
        self._eval_thread = QThread(self)
        self._eval_worker = _EvalWorker(
            df=df.copy(),
            spec_config=self._material_dataset.spec_config if self._material_dataset is not None else None,
            config=dict(self._config),
            scenario_a=scenario_a,
            scenario_b=scenario_b,
        )
        self._eval_worker.moveToThread(self._eval_thread)
        self._eval_thread.started.connect(self._eval_worker.run)
        self._eval_worker.finished.connect(self._on_eval_finished)
        self._eval_worker.failed.connect(self._on_eval_failed)
        self._eval_worker.finished.connect(self._eval_thread.quit)
        self._eval_worker.failed.connect(self._eval_thread.quit)
        self._eval_thread.finished.connect(self._cleanup_eval_worker)
        self._eval_thread.start()

    def _on_eval_finished(self, result: dict[str, Any]) -> None:
        self._render_eval_result(result)
        self.eval_status_label.setText("Completed.")
        self.evaluate_button.setEnabled(True)

    def _on_eval_failed(self, message: str) -> None:
        self.interpretation.setPlainText(f"Evaluation failed: {message}")
        self.eval_status_label.setText("Failed.")
        self.evaluate_button.setEnabled(True)

    def _cleanup_eval_worker(self) -> None:
        if self._eval_worker is not None:
            self._eval_worker.deleteLater()
        self._eval_worker = None
        if self._eval_thread is not None:
            self._eval_thread.deleteLater()
        self._eval_thread = None

    def _render_eval_result(self, result: dict[str, Any]) -> None:
        a_out = result.get("scenario_a", {})
        b_out = result.get("scenario_b", {})
        if "error" in a_out or "error" in b_out:
            msg = f"Scenario A: {a_out.get('error', 'ok')}\nScenario B: {b_out.get('error', 'ok')}"
            self.interpretation.setPlainText(msg)
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            self.ax.clear()
            if self.canvas is not None:
                self.canvas.draw_idle()
            return

        metrics = [
            ("Train rows", float(a_out["row_summary"]["n_train"]), float(b_out["row_summary"]["n_train"])),
            ("Test rows", float(a_out["row_summary"]["n_test"]), float(b_out["row_summary"]["n_test"])),
            ("History target count", float(len(a_out["config"]["history_target_ids"])), float(len(b_out["config"]["history_target_ids"]))),
            ("RS MAE", a_out["metrics"].get("rs_mae"), b_out["metrics"].get("rs_mae")),
            ("RS RMSE", a_out["metrics"].get("rs_rmse"), b_out["metrics"].get("rs_rmse")),
            ("Thickness MAE", a_out["metrics"].get("thickness_mae"), b_out["metrics"].get("thickness_mae")),
            ("Thickness RMSE", a_out["metrics"].get("thickness_rmse"), b_out["metrics"].get("thickness_rmse")),
            ("RSU MAE", a_out["metrics"].get("rsu_mae"), b_out["metrics"].get("rsu_mae")),
            ("RSU RMSE", a_out["metrics"].get("rsu_rmse"), b_out["metrics"].get("rsu_rmse")),
            ("Spec pass accuracy", a_out["metrics"].get("spec_pass_accuracy"), b_out["metrics"].get("spec_pass_accuracy")),
            (
                "Rec improvement rate",
                a_out["recommendation_proxy"].get("improvement_rate"),
                b_out["recommendation_proxy"].get("improvement_rate"),
            ),
        ]
        self._fill_metrics_table(metrics)
        self._draw_metrics_plot(metrics)
        self.interpretation.setPlainText(
            str(result.get("interpretation", "No interpretation available."))
            + "\n\n"
            + f"Scenario A confidence: {a_out.get('confidence_summary', {}).get('level', 'unknown')}\n"
            + f"Scenario B confidence: {b_out.get('confidence_summary', {}).get('level', 'unknown')}"
        )

    def _fill_metrics_table(self, metrics: list[tuple[str, Optional[float], Optional[float]]]) -> None:
        self.metrics_table.clear()
        self.metrics_table.setRowCount(len(metrics))
        self.metrics_table.setColumnCount(4)
        headers = ["Metric", "Scenario A", "Scenario B", "B - A"]
        for ci, header in enumerate(headers):
            self.metrics_table.setHorizontalHeaderItem(ci, QTableWidgetItem(header))
        for ri, (name, a_val, b_val) in enumerate(metrics):
            self.metrics_table.setItem(ri, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(ri, 1, QTableWidgetItem(self._fmt(a_val)))
            self.metrics_table.setItem(ri, 2, QTableWidgetItem(self._fmt(b_val)))
            delta = (b_val - a_val) if a_val is not None and b_val is not None else None
            self.metrics_table.setItem(ri, 3, QTableWidgetItem(self._fmt(delta)))
        self.metrics_table.resizeRowsToContents()

    def _draw_metrics_plot(self, metrics: list[tuple[str, Optional[float], Optional[float]]]) -> None:
        self.ax.clear()
        plot_metrics = [m for m in metrics if m[0] in {"RS MAE", "Thickness MAE", "RSU MAE"}]
        labels = [m[0] for m in plot_metrics]
        a_vals = [float(m[1]) if m[1] is not None else 0.0 for m in plot_metrics]
        b_vals = [float(m[2]) if m[2] is not None else 0.0 for m in plot_metrics]
        x = list(range(len(labels)))
        width = 0.35
        self.ax.bar([v - width / 2 for v in x], a_vals, width=width, label="Scenario A")
        self.ax.bar([v + width / 2 for v in x], b_vals, width=width, label="Scenario B")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=15)
        self.ax.set_ylabel("Error")
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(fontsize=8, loc="best")
        self.figure.tight_layout()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _fmt(self, val: Optional[float]) -> str:
        if val is None:
            return "-"
        return f"{float(val):.6g}"

