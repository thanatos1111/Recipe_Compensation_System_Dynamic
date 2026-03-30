"""
Model training and evaluation panel.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.evaluation_mode import ScenarioConfig, evaluate_data_sufficiency_scenarios
from core.feature_engineering import build_feature_matrix, get_feature_schema
from core.instance_correction import fit_instance_bias, summarize_recent_residuals
from core.model_explanations import TrainingExplanation, build_training_explanation
from core.pipeline_inspection import build_processing_view_data
from core.benchmarking import (
    MODEL_BUNDLE_PRESETS,
    flatten_benchmark_suite_folds,
    rank_benchmark_results,
    run_prediction_benchmark,
)
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
        self._processing_stages: list[dict[str, Any]] = []
        self._last_training_artifacts: Optional[Any] = None
        self._training_target_ids: tuple[str, ...] = ()
        self._model_plot_hover_items: dict[str, list[tuple[Any, float, float, str]]] = {"rs": [], "thickness": [], "rsu": []}
        self._cutoff_by_target: dict[str, float] = {}
        self._last_benchmark_suite_results: Optional[Any] = None
        self._last_benchmark_ranked: list[dict[str, Any]] = []
        self._last_benchmark_best_bundle: Optional[str] = None

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
        self.train_targets_expand_btn = QToolButton()
        self.train_targets_expand_btn.setText("Training target selection (default: all)")
        self.train_targets_expand_btn.setCheckable(True)
        self.train_targets_expand_btn.setChecked(False)
        self.train_targets_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.train_targets_expand_btn.toggled.connect(self._toggle_training_targets_list)
        training_layout.addWidget(self.train_targets_expand_btn)
        self.training_targets_list = QListWidget()
        self.training_targets_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.training_targets_list.itemSelectionChanged.connect(self._refresh_model_pipeline_view)
        self.training_targets_list.setVisible(False)
        self.training_targets_list.setMinimumHeight(110)
        training_layout.addWidget(self.training_targets_list)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        training_layout.addWidget(self.output)
        training_layout.setStretchFactor(self.output, 1)

        self.training_help_expand_btn = QToolButton()
        self.training_help_expand_btn.setText("Training metric explanations")
        self.training_help_expand_btn.setCheckable(True)
        self.training_help_expand_btn.setChecked(False)
        self.training_help_expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.training_help_expand_btn.toggled.connect(self._toggle_training_help)
        training_layout.addWidget(self.training_help_expand_btn)

        self.training_help_text = QTextEdit()
        self.training_help_text.setReadOnly(True)
        self.training_help_text.setVisible(False)
        self.training_help_text.setToolTip(
            "Dynamic explanation for training rows, spec counts, features, model type, validation, errors, "
            "feature influence, residuals, confidence, and warnings."
        )
        training_layout.addWidget(self.training_help_text)
        training_layout.setStretchFactor(self.training_help_text, 1)
        training_page.setLayout(training_layout)
        self.section_tabs.addTab(training_page, "1) Training")

        # Section 2: model training pipeline introspection.
        model_pipeline_page = QWidget()
        model_pipeline_layout = QVBoxLayout()
        self.model_pipeline_header = QLabel("Training pipeline: input -> feature matrix -> targets -> fit -> evaluation -> confidence")
        self.model_pipeline_header.setWordWrap(True)
        self.model_pipeline_header.setMaximumHeight(24)
        model_pipeline_layout.addWidget(self.model_pipeline_header)

        model_pipeline_split = QSplitter()
        model_pipeline_split.setOrientation(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        self.model_pipeline_stage_table = QTableWidget()
        self.model_pipeline_stage_table.setSelectionBehavior(self.model_pipeline_stage_table.SelectionBehavior.SelectRows)
        self.model_pipeline_stage_table.setSelectionMode(self.model_pipeline_stage_table.SelectionMode.SingleSelection)
        self.model_pipeline_stage_table.itemSelectionChanged.connect(self._on_model_pipeline_stage_selected)
        left_layout.addWidget(self.model_pipeline_stage_table)
        self.model_pipeline_metrics_table = QTableWidget()
        left_layout.addWidget(self.model_pipeline_metrics_table)
        left_layout.setStretchFactor(self.model_pipeline_stage_table, 3)
        left_layout.setStretchFactor(self.model_pipeline_metrics_table, 2)
        left_panel.setLayout(left_layout)
        model_pipeline_split.addWidget(left_panel)

        right_model_panel = QWidget()
        right_model_layout = QVBoxLayout()
        self.model_pipeline_plot_mode_tabs = QTabWidget()

        self.model_pipeline_plot_tabs = QTabWidget()
        self.model_pipeline_canvases: dict[str, Any] = {}
        self.model_pipeline_figures: dict[str, Any] = {}
        self.model_pipeline_axes: dict[str, list[Any]] = {}
        for resp, title in [("rs", "RS"), ("thickness", "Thickness"), ("rsu", "RSU")]:
            tab = QWidget()
            tab_layout = QVBoxLayout()
            fig = Figure(figsize=(9, 5.4))
            axes = [
                fig.add_subplot(231),
                fig.add_subplot(232),
                fig.add_subplot(233),
                fig.add_subplot(234),
                fig.add_subplot(235),
                fig.add_subplot(236),
            ]
            self.model_pipeline_figures[resp] = fig
            self.model_pipeline_axes[resp] = axes
            if FigureCanvas is not None:
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(300)
                canvas.setMaximumHeight(430)
                canvas.mpl_connect("motion_notify_event", lambda event, r=resp: self._on_model_plot_hover(event, r))
                tab_layout.addWidget(canvas)
                self.model_pipeline_canvases[resp] = canvas
            guide = QLabel(
                "How to read: (1) Stage rows show data kept at each training step. "
                "(2) Missing ratio highlights weak features. "
                "(3) Signal plot shows measured response across all targets. "
                "(4) Parity closer to diagonal is better. "
                "(5) Residuals centered near 0 indicate lower bias. "
                "(6) Feature influence ranks important features."
            )
            guide.setWordWrap(True)
            guide.setMaximumHeight(42)
            tab_layout.addWidget(guide)
            tab.setLayout(tab_layout)
            self.model_pipeline_plot_tabs.addTab(tab, title)
        self.model_pipeline_plot_mode_tabs.addTab(self.model_pipeline_plot_tabs, "By Response")

        self.model_stage_plot_tabs = QTabWidget()
        self.model_stage_figures: dict[str, Any] = {}
        self.model_stage_canvases: dict[str, Any] = {}
        self.model_stage_axes: dict[str, Any] = {}
        for stage_key, stage_title in [
            ("feature_matrix", "Feature Matrix"),
            ("target_extraction", "Target Extraction"),
            ("model_fitting", "Model Fitting"),
            ("time_aware_eval", "Time-Aware Eval"),
            ("confidence", "Confidence Synthesis"),
        ]:
            tab = QWidget()
            tab_layout = QVBoxLayout()
            fig = Figure(figsize=(8.8, 4.4))
            ax = fig.add_subplot(111)
            self.model_stage_figures[stage_key] = fig
            self.model_stage_axes[stage_key] = ax
            if FigureCanvas is not None:
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(280)
                tab_layout.addWidget(canvas)
                self.model_stage_canvases[stage_key] = canvas
            tab.setLayout(tab_layout)
            self.model_stage_plot_tabs.addTab(tab, stage_title)
        self.model_pipeline_plot_mode_tabs.addTab(self.model_stage_plot_tabs, "By Stage")
        right_model_layout.addWidget(self.model_pipeline_plot_mode_tabs)

        self.model_pipeline_stage_details = QTextEdit()
        self.model_pipeline_stage_details.setReadOnly(True)
        right_model_layout.addWidget(self.model_pipeline_stage_details)
        right_model_layout.setStretchFactor(self.model_pipeline_plot_mode_tabs, 4)
        right_model_layout.setStretchFactor(self.model_pipeline_stage_details, 2)
        right_model_panel.setLayout(right_model_layout)
        model_pipeline_split.addWidget(right_model_panel)
        model_pipeline_split.setSizes([420, 700])
        model_pipeline_layout.addWidget(model_pipeline_split)
        model_pipeline_page.setLayout(model_pipeline_layout)
        self.section_tabs.addTab(model_pipeline_page, "2) Model Pipeline")

        # Section 3: data processing view (read-only).
        processing_page = QWidget()
        processing_layout = QVBoxLayout()
        self.pipeline_diagram_label = QLabel(
            "Pipeline: raw -> normalized -> grouped targets -> derived columns -> in-spec labels -> trend rows -> training rows -> excluded rows"
        )
        self.pipeline_diagram_label.setWordWrap(True)
        self.pipeline_diagram_label.setMaximumHeight(28)
        processing_layout.addWidget(self.pipeline_diagram_label)

        self.feature_list_label = QLabel("Features: -")
        self.feature_list_label.setWordWrap(True)
        self.feature_list_label.setMaximumHeight(42)
        processing_layout.addWidget(self.feature_list_label)

        self.processing_split = QSplitter()
        self.processing_split.setOrientation(Qt.Orientation.Horizontal)
        self.processing_split.setChildrenCollapsible(False)

        self.stage_summary_table = QTableWidget()
        self.stage_summary_table.setMinimumWidth(320)
        self.stage_summary_table.setMinimumHeight(0)
        self.stage_summary_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.stage_summary_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stage_summary_table.setVerticalScrollMode(self.stage_summary_table.ScrollMode.ScrollPerPixel)
        self.stage_summary_table.setSelectionBehavior(self.stage_summary_table.SelectionBehavior.SelectRows)
        self.stage_summary_table.setSelectionMode(self.stage_summary_table.SelectionMode.SingleSelection)
        self.stage_summary_table.itemSelectionChanged.connect(self._on_stage_selection_changed)
        processing_split_header = self.stage_summary_table.horizontalHeader()
        processing_split_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.processing_split.addWidget(self.stage_summary_table)

        right_wrap = QWidget()
        right_layout = QVBoxLayout()
        self.stage_preview_title = QLabel("Preview: -")
        right_layout.addWidget(self.stage_preview_title)
        self.stage_preview_table = QTableWidget()
        self.stage_preview_table.setMinimumHeight(0)
        self.stage_preview_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.stage_preview_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stage_preview_table.setVerticalScrollMode(self.stage_preview_table.ScrollMode.ScrollPerPixel)
        self.stage_preview_table.setWordWrap(False)
        self.stage_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.stage_preview_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        right_layout.addWidget(self.stage_preview_table)
        right_layout.setStretchFactor(self.stage_preview_table, 1)
        right_wrap.setLayout(right_layout)
        self.processing_split.addWidget(right_wrap)
        self.processing_split.setSizes([430, 650])
        self.processing_split.setStretchFactor(0, 2)
        self.processing_split.setStretchFactor(1, 3)
        processing_layout.addWidget(self.processing_split)
        processing_layout.setStretchFactor(self.processing_split, 1)
        processing_page.setLayout(processing_layout)
        self.section_tabs.addTab(processing_page, "3) Data Processing View")

        # Section 4: scenario evaluation only.
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
        self.cutoff_spin.setKeyboardTracking(False)
        self.cutoff_spin.setValue(0.0)
        self.cutoff_spin.editingFinished.connect(self._on_cutoff_edit_finished)
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
        self.section_tabs.addTab(eval_page, "4) Scenario Evaluation")

        # Section 5: benchmark runner.
        benchmark_page = QWidget()
        benchmark_page_layout = QVBoxLayout()

        # The benchmark content can be tall (controls + tables + chart + text),
        # so wrap it in a scroll area to avoid out-of-screen overlap.
        benchmark_scroll = QScrollArea()
        benchmark_scroll.setWidgetResizable(True)
        benchmark_content = QWidget()
        benchmark_layout = QVBoxLayout()
        benchmark_content.setLayout(benchmark_layout)
        benchmark_scroll.setWidget(benchmark_content)
        benchmark_page_layout.addWidget(benchmark_scroll)

        benchmark_controls_box = QGroupBox("Benchmark setup")
        benchmark_form = QFormLayout()
        benchmark_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        benchmark_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        benchmark_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.bench_split_mode_combo = QComboBox()
        self.bench_split_mode_combo.addItems(["forward_chaining", "leave_one_target_out", "active_target_cutoff"])
        self.bench_split_mode_combo.currentTextChanged.connect(self._on_benchmark_split_mode_changed)
        benchmark_form.addRow("Split mode:", self.bench_split_mode_combo)

        self.bench_bundle_list = QListWidget()
        self.bench_bundle_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for bundle_name in MODEL_BUNDLE_PRESETS.keys():
            self.bench_bundle_list.addItem(QListWidgetItem(bundle_name))
        self.bench_bundle_list.setMinimumHeight(90)
        benchmark_form.addRow("Model bundles:", self.bench_bundle_list)

        # Active-target parameters (only visible for active_target_cutoff split mode).
        self.bench_active_cutoff_box = QWidget()
        active_cutoff_layout = QHBoxLayout()
        self.bench_active_target_combo = QComboBox()
        self.bench_cutoff_spin = QDoubleSpinBox()
        self.bench_cutoff_spin.setRange(0.0, 1e9)
        self.bench_cutoff_spin.setDecimals(6)
        self.bench_cutoff_spin.setKeyboardTracking(False)
        self.bench_cutoff_spin.setValue(0.0)
        active_cutoff_layout.addWidget(QLabel("Active target:"))
        active_cutoff_layout.addWidget(self.bench_active_target_combo)
        active_cutoff_layout.addSpacing(12)
        active_cutoff_layout.addWidget(QLabel("Cutoff:"))
        active_cutoff_layout.addWidget(self.bench_cutoff_spin)
        self.bench_active_cutoff_box.setLayout(active_cutoff_layout)
        benchmark_form.addRow("Active cutoff params:", self.bench_active_cutoff_box)

        self.bench_chart_metric_combo = QComboBox()
        self.bench_chart_metric_combo.addItems(["RS MAE", "Thickness MAE", "RSU MAE", "Spec pass accuracy"])
        benchmark_form.addRow("Chart metric:", self.bench_chart_metric_combo)

        action_row = QHBoxLayout()
        self.bench_run_button = QPushButton("Run benchmark")
        self.bench_run_button.clicked.connect(self._benchmark_run_clicked)
        self.bench_status_label = QLabel("")
        action_row.addWidget(self.bench_run_button)
        action_row.addWidget(self.bench_status_label)
        action_row.addStretch(1)

        benchmark_export_row = QHBoxLayout()
        self.bench_export_summary_button = QPushButton("Export summary CSV")
        self.bench_export_summary_button.clicked.connect(self._benchmark_export_summary_csv)
        self.bench_export_folds_button = QPushButton("Export folds CSV")
        self.bench_export_folds_button.clicked.connect(self._benchmark_export_folds_csv)
        benchmark_export_row.addWidget(self.bench_export_summary_button)
        benchmark_export_row.addWidget(self.bench_export_folds_button)

        action_row2 = QHBoxLayout()
        self.bench_adopt_winner_button = QPushButton("Adopt winner models (no retrain)")
        self.bench_adopt_winner_button.clicked.connect(self._benchmark_adopt_winner_clicked)
        action_row2.addWidget(self.bench_adopt_winner_button)
        action_row2.addStretch(1)

        benchmark_form.addRow(action_row)
        benchmark_form.addRow(benchmark_export_row)
        benchmark_form.addRow(action_row2)

        benchmark_controls_box.setLayout(benchmark_form)
        benchmark_layout.addWidget(benchmark_controls_box)

        self.bench_summary_table = QTableWidget()
        self.bench_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bench_summary_table.setMinimumHeight(120)
        benchmark_layout.addWidget(self.bench_summary_table)

        self.bench_fold_table = QTableWidget()
        self.bench_fold_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bench_fold_table.setMinimumHeight(180)
        benchmark_layout.addWidget(self.bench_fold_table)

        bench_bottom_split = QSplitter()
        bench_bottom_split.setOrientation(Qt.Orientation.Horizontal)

        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        self.bench_fig = Figure(figsize=(6, 4))
        self.bench_ax = self.bench_fig.add_subplot(111)
        if FigureCanvas is not None:
            self.bench_canvas = FigureCanvas(self.bench_fig)
            self.bench_canvas.setMinimumHeight(220)
            chart_layout.addWidget(self.bench_canvas)
        else:
            self.bench_canvas = None
        chart_widget.setLayout(chart_layout)
        bench_bottom_split.addWidget(chart_widget)

        self.bench_best_explanation = QTextEdit()
        self.bench_best_explanation.setReadOnly(True)
        bench_bottom_split.addWidget(self.bench_best_explanation)
        bench_bottom_split.setSizes([420, 420])

        benchmark_layout.addWidget(bench_bottom_split)
        benchmark_page.setLayout(benchmark_page_layout)
        self.section_tabs.addTab(benchmark_page, "5) Benchmark")
        self._on_benchmark_split_mode_changed(self.bench_split_mode_combo.currentText())

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
        self._set_target_list_items(self.training_targets_list, target_ids)
        self._select_all_in_list(self.training_targets_list)
        self._refresh_history_target_lists(active_target_id)
        self._init_cutoff(material_dataset, active_target_id)
        self._refresh_history_selection_enabled_state()
        self._refresh_training_explanation()
        self._refresh_processing_view()
        self._refresh_model_pipeline_view()
        self._refresh_benchmark_active_target_combo(active_target_id=active_target_id, target_ids=target_ids)

    def _train_clicked(self) -> None:
        if self._material_dataset is None:
            self.output.setPlainText("No material selected.")
            return

        df = self._material_dataset.all_records
        if df is None or df.empty:
            self.output.setPlainText("No rows available for training.")
            return
        selected_targets = self._selected_targets_from_widget(self.training_targets_list)
        if selected_targets:
            train_df = df[df["target_id"].astype(str).isin(selected_targets)].copy()
        else:
            train_df = df.copy()
        if train_df.empty:
            self.output.setPlainText("No rows left after selected training targets filter.")
            return

        # Train per-material models using all rows (including out-of-spec).
        artifacts = train_material_models(
            train_df,
            config=self._config,
            spec_config=self._material_dataset.spec_config,
        )
        self._last_training_artifacts = artifacts
        self._training_target_ids = tuple(selected_targets) if selected_targets else tuple(self._ordered_target_ids)
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
            f"- selected_training_targets: {list(self._training_target_ids)}\n"
            f"- target_instances: {artifacts.train_summary.get('target_instance_count')}\n"
            "\nMetrics (leakage-free benchmark):\n"
            f"- rs_mae: {artifacts.metrics.get('rs_mae')}\n"
            f"- thickness_mae: {artifacts.metrics.get('thickness_mae')}\n"
            f"- rsu_mae: {artifacts.metrics.get('rsu_mae')}\n"
            "\nConfidence:\n"
            f"- level: {artifacts.confidence_summary.get('level')}\n"
            f"- notes: {artifacts.confidence_summary.get('notes')}\n"
            f"{correction_text}"
        )
        self._refresh_training_explanation(train_df=train_df)
        self._refresh_model_pipeline_view()

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

    def _select_all_in_list(self, widget: QListWidget) -> None:
        for i in range(widget.count()):
            widget.item(i).setSelected(True)

    def _toggle_training_targets_list(self, expanded: bool) -> None:
        self.training_targets_list.setVisible(expanded)
        self.train_targets_expand_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._refresh_training_explanation()

    def _toggle_training_help(self, expanded: bool) -> None:
        self.training_help_text.setVisible(expanded)
        self.training_help_expand_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _on_cutoff_edit_finished(self) -> None:
        self.cutoff_spin.interpretText()
        active_target_id = self.eval_active_target_combo.currentText().strip()
        if active_target_id:
            self._cutoff_by_target[active_target_id] = float(self.cutoff_spin.value())

    def _on_active_target_changed(self, target_id: str) -> None:
        if self._material_dataset is None:
            return
        self._active_target_id = target_id
        self.training_active_target_label.setText(target_id)
        self._refresh_history_target_lists(target_id)
        self._init_cutoff(self._material_dataset, target_id)
        self._refresh_training_explanation()
        self._refresh_processing_view()
        self._refresh_model_pipeline_view()

    def _refresh_model_pipeline_view(self) -> None:
        if self._material_dataset is None:
            return
        df_all = self._material_dataset.all_records
        selected_targets = self._selected_targets_from_widget(self.training_targets_list)
        if df_all is not None and not df_all.empty and selected_targets:
            df = df_all[df_all["target_id"].astype(str).isin(selected_targets)].copy()
        else:
            df = df_all
        if df is None or df.empty:
            self.model_pipeline_stage_table.setRowCount(0)
            self.model_pipeline_metrics_table.setRowCount(0)
            self.model_pipeline_metrics_table.setColumnCount(0)
            self.model_pipeline_stage_details.setPlainText("No rows available.")
            self._refresh_training_explanation()
            return

        feature_config = self._config.get("feature_config", {})
        schema = get_feature_schema(feature_config)
        X = build_feature_matrix(df, feature_config)
        rs_valid = int(pd.to_numeric(df.get("rs"), errors="coerce").notna().sum()) if "rs" in df.columns else 0
        thickness_valid = int(pd.to_numeric(df.get("thickness"), errors="coerce").notna().sum()) if "thickness" in df.columns else 0
        rsu_valid = int(pd.to_numeric(df.get("rsu"), errors="coerce").notna().sum()) if "rsu" in df.columns else 0
        trained = self._last_training_artifacts is not None

        stage_rows = [
            ("Input rows", "ready", f"rows={len(df)}; targets={df['target_id'].nunique() if 'target_id' in df.columns else 0}"),
            ("Feature matrix", "ready", f"rows={len(X)}; cols={len(X.columns)}"),
            ("Target extraction", "ready", f"rs={rs_valid}, thickness={thickness_valid}, rsu={rsu_valid}"),
            ("Model fitting", "trained" if trained else "pending", "train_button required"),
            ("Leakage-free benchmark", "trained" if trained else "pending", "metrics available after training"),
            ("Confidence synthesis", "trained" if trained else "pending", "summary available after training"),
        ]
        self._render_model_pipeline_stages(stage_rows)
        self._render_model_pipeline_metrics(schema)
        self._render_model_pipeline_plots(df, X)
        self._refresh_training_explanation(train_df=df)
        if stage_rows:
            self.model_pipeline_stage_table.selectRow(0)

    def _current_training_dataframe(self) -> Optional[pd.DataFrame]:
        if self._material_dataset is None:
            return None
        df_all = self._material_dataset.all_records
        if df_all is None or df_all.empty:
            return df_all
        selected_targets = self._selected_targets_from_widget(self.training_targets_list)
        if selected_targets:
            return df_all[df_all["target_id"].astype(str).isin(selected_targets)].copy()
        return df_all.copy()

    def _refresh_training_explanation(self, *, train_df: Optional[pd.DataFrame] = None) -> None:
        if train_df is None:
            train_df = self._current_training_dataframe()

        feature_config = self._config.get("feature_config", {})
        schema = get_feature_schema(feature_config)
        explanation = build_training_explanation(
            train_df=train_df,
            artifacts=self._last_training_artifacts,
            active_target_id=self._active_target_id,
            feature_schema={"numeric": list(schema.numeric_features), "categorical": list(schema.categorical_features)},
        )
        self._render_training_explanation(explanation)

    def _render_training_explanation(self, explanation: TrainingExplanation) -> None:
        warning_text = "\n".join(f"- {w}" for w in explanation.warnings) if explanation.warnings else "- none"
        lines = [
            "Model training explanation",
            "",
            explanation.summary,
            "",
            "Metric interpretation:",
        ]
        for item in explanation.items:
            lines.append(f"- {item.label}: {item.value}")
            lines.append(f"  Meaning: {item.detail}")
            lines.append(f"  Help: {item.tooltip}")
        lines.extend(["", "Warnings:", warning_text])
        self.training_help_text.setPlainText("\n".join(lines))
        self.training_help_expand_btn.setToolTip(explanation.summary)
        self.training_help_text.setToolTip(explanation.summary)

    def _render_model_pipeline_stages(self, rows: list[tuple[str, str, str]]) -> None:
        self.model_pipeline_stage_table.clear()
        self.model_pipeline_stage_table.setRowCount(len(rows))
        self.model_pipeline_stage_table.setColumnCount(3)
        headers = ["Stage", "Status", "Notes"]
        for ci, h in enumerate(headers):
            self.model_pipeline_stage_table.setHorizontalHeaderItem(ci, QTableWidgetItem(h))
        for ri, (name, status, notes) in enumerate(rows):
            self.model_pipeline_stage_table.setItem(ri, 0, QTableWidgetItem(name))
            self.model_pipeline_stage_table.setItem(ri, 1, QTableWidgetItem(status))
            self.model_pipeline_stage_table.setItem(ri, 2, QTableWidgetItem(notes))
        self.model_pipeline_stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _render_model_pipeline_metrics(self, schema: Any) -> None:
        self.model_pipeline_metrics_table.clear()
        metrics_rows: list[tuple[str, str]] = []
        metrics_rows.append(("numeric_features", ", ".join(schema.numeric_features)))
        metrics_rows.append(("categorical_features", ", ".join(schema.categorical_features)))
        if self._last_training_artifacts is not None:
            artifacts = self._last_training_artifacts
            for k, v in (artifacts.train_summary or {}).items():
                metrics_rows.append((f"train_summary.{k}", str(v)))
            for k, v in (artifacts.metrics or {}).items():
                metrics_rows.append((f"metrics.{k}", str(v)))
            for k, v in (artifacts.confidence_summary or {}).items():
                metrics_rows.append((f"confidence.{k}", str(v)))
        else:
            metrics_rows.append(("status", "Train models to populate internal metrics."))

        self.model_pipeline_metrics_table.setRowCount(len(metrics_rows))
        self.model_pipeline_metrics_table.setColumnCount(2)
        self.model_pipeline_metrics_table.setHorizontalHeaderItem(0, QTableWidgetItem("Key"))
        self.model_pipeline_metrics_table.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        for ri, (k, v) in enumerate(metrics_rows):
            self.model_pipeline_metrics_table.setItem(ri, 0, QTableWidgetItem(k))
            self.model_pipeline_metrics_table.setItem(ri, 1, QTableWidgetItem(v))
        self.model_pipeline_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _on_model_pipeline_stage_selected(self) -> None:
        selected = self.model_pipeline_stage_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        stage_name_item = self.model_pipeline_stage_table.item(row, 0)
        status_item = self.model_pipeline_stage_table.item(row, 1)
        note_item = self.model_pipeline_stage_table.item(row, 2)
        stage_name = stage_name_item.text() if stage_name_item is not None else "-"
        status = status_item.text() if status_item is not None else "-"
        note = note_item.text() if note_item is not None else "-"
        self.model_pipeline_stage_details.setPlainText(
            f"Stage: {stage_name}\nStatus: {status}\n\nDetails:\n{note}\n\n"
            "This view is read-only and intended for understanding how the training pipeline is assembled."
        )

    def _render_model_pipeline_plots(self, df: pd.DataFrame, X: pd.DataFrame) -> None:
        for resp in ["rs", "thickness", "rsu"]:
            canvas = self.model_pipeline_canvases.get(resp)
            if canvas is None:
                continue
            axes = self.model_pipeline_axes[resp]
            self._model_plot_hover_items[resp] = []
            for ax in axes:
                ax.clear()
                ax.set_axis_on()

            self._draw_response_pipeline_plots(resp, df, X, axes)
            self.model_pipeline_figures[resp].tight_layout()
            canvas.draw_idle()
        self._render_stage_pipeline_plots(df, X)

    def _render_stage_pipeline_plots(self, df: pd.DataFrame, X: pd.DataFrame) -> None:
        # 1) Feature matrix diagnostics.
        ax = self.model_stage_axes.get("feature_matrix")
        canvas = self.model_stage_canvases.get("feature_matrix")
        if ax is not None and canvas is not None:
            ax.clear()
            if X is None or X.empty:
                ax.text(0.5, 0.5, "No feature matrix", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                miss = X.isna().mean().sort_values(ascending=False).head(15)
                ax.barh(list(reversed([str(v) for v in miss.index])), list(reversed([float(v) for v in miss.values])), color="#f28e2b")
                ax.set_xlim(0, 1)
                ax.set_title("Feature missing ratio (top 15)")
                ax.set_xlabel("missing ratio")
                ax.grid(True, alpha=0.2, axis="x")
            self.model_stage_figures["feature_matrix"].tight_layout()
            canvas.draw_idle()

        # 2) Target extraction diagnostics.
        ax = self.model_stage_axes.get("target_extraction")
        canvas = self.model_stage_canvases.get("target_extraction")
        if ax is not None and canvas is not None:
            ax.clear()
            total = len(df)
            rs_valid = int(pd.to_numeric(df.get("rs"), errors="coerce").notna().sum()) if "rs" in df.columns else 0
            th_valid = int(pd.to_numeric(df.get("thickness"), errors="coerce").notna().sum()) if "thickness" in df.columns else 0
            rsu_valid = int(pd.to_numeric(df.get("rsu"), errors="coerce").notna().sum()) if "rsu" in df.columns else 0
            cats = ["rows_total", "rs_valid", "thickness_valid", "rsu_valid"]
            vals = [total, rs_valid, th_valid, rsu_valid]
            ax.bar(cats, vals, color=["#4c78a8", "#59a14f", "#e15759", "#76b7b2"])
            ax.set_title("Target extraction validity")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True, alpha=0.2, axis="y")
            self.model_stage_figures["target_extraction"].tight_layout()
            canvas.draw_idle()

        # 3) Model fitting diagnostics.
        ax = self.model_stage_axes.get("model_fitting")
        canvas = self.model_stage_canvases.get("model_fitting")
        if ax is not None and canvas is not None:
            ax.clear()
            if self._last_training_artifacts is None:
                ax.text(0.5, 0.5, "Train model first", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                metrics = self._last_training_artifacts.metrics or {}
                cats = ["rs_mae", "thickness_mae", "rsu_mae"]
                vals = [float(metrics.get("rs_mae") or 0.0), float(metrics.get("thickness_mae") or 0.0), float(metrics.get("rsu_mae") or 0.0)]
                ax.bar(cats, vals, color=["#59a14f", "#e15759", "#76b7b2"])
                ax.set_title("Model fitting quality (MAE)")
                ax.grid(True, alpha=0.2, axis="y")
            self.model_stage_figures["model_fitting"].tight_layout()
            canvas.draw_idle()

        # 4) Benchmark split diagnostics.
        ax = self.model_stage_axes.get("time_aware_eval")
        canvas = self.model_stage_canvases.get("time_aware_eval")
        if ax is not None and canvas is not None:
            ax.clear()
            if self._last_training_artifacts is None:
                ax.text(0.5, 0.5, "Train model first", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                metrics = self._last_training_artifacts.metrics or {}
                n_train = float(metrics.get("n_train") or 0.0)
                n_test = float(metrics.get("n_test") or 0.0)
                n_folds = int(metrics.get("n_folds") or 0)
                st = str(metrics.get("benchmark_split_type") or "forward_chaining")
                ax.bar(["n_train", "n_test"], [n_train, n_test], color=["#4c78a8", "#f28e2b"])
                ax.set_title(f"Benchmark split ({st}, folds={n_folds})")
                ax.grid(True, alpha=0.2, axis="y")
            self.model_stage_figures["time_aware_eval"].tight_layout()
            canvas.draw_idle()

        # 5) Confidence synthesis diagnostics.
        ax = self.model_stage_axes.get("confidence")
        canvas = self.model_stage_canvases.get("confidence")
        if ax is not None and canvas is not None:
            ax.clear()
            if self._last_training_artifacts is None:
                ax.text(0.5, 0.5, "Train model first", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ts = self._last_training_artifacts.train_summary or {}
                conf = self._last_training_artifacts.confidence_summary or {}
                row_count = float(ts.get("row_count") or 0.0)
                target_count = float(ts.get("target_instance_count") or 0.0)
                ax.bar(["row_count", "target_count"], [row_count, target_count], color=["#4c78a8", "#59a14f"])
                level = str(conf.get("level", "unknown"))
                notes = conf.get("notes", [])
                ax.set_title(f"Confidence synthesis: {level}")
                ax.text(
                    0.02,
                    0.95,
                    f"notes={notes}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                ax.grid(True, alpha=0.2, axis="y")
            self.model_stage_figures["confidence"].tight_layout()
            canvas.draw_idle()

    def _draw_response_pipeline_plots(self, response: str, df: pd.DataFrame, X: pd.DataFrame, axes: list[Any]) -> None:
        target_col = response
        model_key = f"{response}_model" if response != "thickness" else "thickness_model"
        label = "RS" if response == "rs" else ("Thickness" if response == "thickness" else "RSU")

        # Plot 1: pipeline stage row counts.
        ax = axes[0]
        stage_names = ["input", "features", "rs_valid", "thk_valid", "rsu_valid"]
        rs_valid = int(pd.to_numeric(df.get("rs"), errors="coerce").notna().sum()) if "rs" in df.columns else 0
        th_valid = int(pd.to_numeric(df.get("thickness"), errors="coerce").notna().sum()) if "thickness" in df.columns else 0
        rsu_valid = int(pd.to_numeric(df.get("rsu"), errors="coerce").notna().sum()) if "rsu" in df.columns else 0
        vals = [len(df), len(X), rs_valid, th_valid, rsu_valid]
        bars = ax.bar(stage_names, vals, color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"])
        ax.set_title(f"1) Stage rows ({label})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.2, axis="y")
        for i, b in enumerate(bars):
            self._add_hover_item(response, ax, float(i), float(vals[i]), f"{stage_names[i]}: {vals[i]}")

        # Plot 2: feature matrix missing ratio.
        ax = axes[1]
        if not X.empty:
            missing_ratio = X.isna().mean().sort_values(ascending=False).head(10)
            if len(missing_ratio) > 0:
                names = [str(c) for c in missing_ratio.index]
                vals_m = [float(v) for v in missing_ratio.values]
                y_pos = list(range(len(names)))
                ax.barh(y_pos, vals_m, color="#f28e2b")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlim(0, 1)
                ax.set_title("2) Feature missing ratio", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2, axis="x")
                for i, v in enumerate(vals_m):
                    self._add_hover_item(response, ax, float(v), float(i), f"{names[i]} missing={v:.3f}")
            else:
                ax.text(0.5, 0.5, "No features", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No feature matrix", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_axis_off()

        # Plot 3: all-target response signal.
        ax = axes[2]
        if "target_id" in df.columns and "lifetime" in df.columns and target_col in df.columns:
            plot_df = df.copy()
            plot_df["target_id"] = plot_df["target_id"].astype(str)
            target_order = sorted(plot_df["target_id"].dropna().unique().tolist())
            for tid in target_order[:8]:
                sub = plot_df[plot_df["target_id"] == tid].sort_values("lifetime")
                x = pd.to_numeric(sub["lifetime"], errors="coerce")
                y = pd.to_numeric(sub[target_col], errors="coerce")
                valid = x.notna() & y.notna()
                x = x[valid]
                y = y[valid]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", linewidth=1.0, markersize=2.0, label=tid)
                for xv, yv in zip(x.head(40), y.head(40)):
                    self._add_hover_item(response, ax, float(xv), float(yv), f"{tid}: lifetime={xv:.4g}, {label}={yv:.4g}")
            ax.set_title(f"3) All-target {label} signal", fontsize=9)
            ax.set_xlabel("lifetime", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if len(target_order) > 0:
                ax.legend(fontsize=6, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                f"No {label} trend available",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
            ax.set_axis_off()

        # Plot 4/5/6: fitting internals (if trained).
        ax_parity = axes[3]
        ax_resid = axes[4]
        ax_imp = axes[5]

        model_obj = getattr(self._last_training_artifacts, model_key, None) if self._last_training_artifacts is not None else None
        if model_obj is None or target_col not in df.columns:
            for a, title in [
                (ax_parity, f"4) {label} parity"),
                (ax_resid, f"5) {label} residuals"),
                (ax_imp, f"6) {label} feature influence"),
            ]:
                a.text(0.5, 0.5, "Train model to view", ha="center", va="center", fontsize=8, transform=a.transAxes)
                a.set_title(title, fontsize=9)
                a.set_axis_off()
            return

        y = pd.to_numeric(df[target_col], errors="coerce")
        valid = y.notna()
        if int(valid.sum()) == 0:
            for a, title in [
                (ax_parity, f"4) {label} parity"),
                (ax_resid, f"5) {label} residuals"),
                (ax_imp, f"6) {label} feature influence"),
            ]:
                a.text(0.5, 0.5, f"No valid {label} targets", ha="center", va="center", fontsize=8, transform=a.transAxes)
                a.set_title(title, fontsize=9)
                a.set_axis_off()
            return

        X_valid = X.loc[valid]
        y_valid = y.loc[valid]
        try:
            y_pred = pd.Series(np.asarray(model_obj.predict(X_valid), dtype=float), index=y_valid.index)
        except Exception:
            y_pred = pd.Series(dtype=float)

        if y_pred.empty:
            for a, title in [
                (ax_parity, f"4) {label} parity"),
                (ax_resid, f"5) {label} residuals"),
                (ax_imp, f"6) {label} feature influence"),
            ]:
                a.text(0.5, 0.5, "Prediction unavailable", ha="center", va="center", fontsize=8, transform=a.transAxes)
                a.set_title(title, fontsize=9)
                a.set_axis_off()
            return

        # Plot 4 parity.
        idx = y_valid.index
        if len(idx) > 450:
            idx = y_valid.sample(n=450, random_state=0).index
        y_s = y_valid.loc[idx]
        p_s = y_pred.loc[idx]
        ax_parity.scatter(y_s, p_s, s=9, alpha=0.6, color="#59a14f")
        lo = float(min(y_s.min(), p_s.min()))
        hi = float(max(y_s.max(), p_s.max()))
        ax_parity.plot([lo, hi], [lo, hi], linestyle="--", color="#888888", linewidth=1)
        ax_parity.set_title(f"4) {label} parity (actual vs pred)", fontsize=9)
        ax_parity.set_xlabel("actual", fontsize=8)
        ax_parity.set_ylabel("pred", fontsize=8)
        ax_parity.tick_params(labelsize=7)
        ax_parity.grid(True, alpha=0.2)
        for xv, yv in zip(y_s.head(70), p_s.head(70)):
            self._add_hover_item(response, ax_parity, float(xv), float(yv), f"actual={xv:.4g}, pred={yv:.4g}")

        # Plot 5 residual histogram.
        resid = (y_valid - y_pred).dropna()
        ax_resid.hist(resid, bins=24, color="#e15759", alpha=0.8)
        ax_resid.axvline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax_resid.set_title(f"5) {label} residual distribution", fontsize=9)
        ax_resid.set_xlabel("actual - pred", fontsize=8)
        ax_resid.tick_params(labelsize=7)
        ax_resid.grid(True, alpha=0.2, axis="y")

        # Plot 6 feature influence.
        plotted = False
        try:
            if hasattr(model_obj, "named_steps"):
                pre = model_obj.named_steps.get("preprocess")
                est = model_obj.named_steps.get("model")
                feat_names = list(pre.get_feature_names_out()) if pre is not None else list(X.columns)
                if hasattr(est, "feature_importances_"):
                    imp = np.asarray(est.feature_importances_, dtype=float)
                    if len(imp) == len(feat_names) and len(imp) > 0:
                        order = np.argsort(imp)[::-1][:10]
                        names = [feat_names[i] for i in order][::-1]
                        vals = [float(imp[i]) for i in order][::-1]
                        ax_imp.barh(names, vals, color="#76b7b2")
                        ax_imp.set_title(f"6) {label} top feature importances", fontsize=9)
                        ax_imp.tick_params(labelsize=7)
                        ax_imp.grid(True, alpha=0.2, axis="x")
                        plotted = True
                elif hasattr(est, "coef_"):
                    coef = np.ravel(np.asarray(est.coef_, dtype=float))
                    if len(coef) == len(feat_names) and len(coef) > 0:
                        abs_coef = np.abs(coef)
                        order = np.argsort(abs_coef)[::-1][:10]
                        names = [feat_names[i] for i in order][::-1]
                        vals = [float(abs_coef[i]) for i in order][::-1]
                        ax_imp.barh(names, vals, color="#76b7b2")
                        ax_imp.set_title(f"6) {label} top |coefficients|", fontsize=9)
                        ax_imp.tick_params(labelsize=7)
                        ax_imp.grid(True, alpha=0.2, axis="x")
                        plotted = True
        except Exception:
            plotted = False

        if not plotted:
            ax_imp.text(0.5, 0.5, "Feature influence unavailable", ha="center", va="center", fontsize=8, transform=ax_imp.transAxes)
            ax_imp.set_title(f"6) {label} feature influence", fontsize=9)
            ax_imp.set_axis_off()

    def _add_hover_item(self, response: str, ax: Any, x: float, y: float, text: str) -> None:
        self._model_plot_hover_items.setdefault(response, []).append((ax, x, y, text))

    def _on_model_plot_hover(self, event: Any, response: str) -> None:
        canvas = self.model_pipeline_canvases.get(response)
        fig = self.model_pipeline_figures.get(response)
        if canvas is None or fig is None:
            return
        if event.inaxes is None or event.x is None or event.y is None:
            if hasattr(canvas, "_hover_annot") and canvas._hover_annot is not None:
                canvas._hover_annot.set_visible(False)
                canvas.draw_idle()
            return

        if not hasattr(canvas, "_hover_annot") or canvas._hover_annot is None:
            canvas._hover_annot = event.inaxes.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                fontsize=7,
            )
            canvas._hover_annot.set_visible(False)

        annot = canvas._hover_annot
        best = None
        best_dist = 14.0
        for ax, x, y, text in self._model_plot_hover_items.get(response, []):
            if ax is not event.inaxes:
                continue
            px, py = ax.transData.transform((x, y))
            dist = float(np.hypot(px - event.x, py - event.y))
            if dist < best_dist:
                best_dist = dist
                best = (x, y, text)
        if best is None:
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()
            return
        bx, by, bt = best
        annot.xy = (bx, by)
        annot.set_text(bt)
        annot.set_visible(True)
        canvas.draw_idle()

    def _refresh_processing_view(self) -> None:
        if self._material_dataset is None:
            return
        active_target_id = self.eval_active_target_combo.currentText().strip()
        if not active_target_id:
            return
        payload = build_processing_view_data(
            material_dataset=self._material_dataset,
            active_target_id=active_target_id,
            config=self._config,
        )
        self._processing_stages = list(payload.get("stages", []))
        feature_list = payload.get("feature_list", [])
        self.feature_list_label.setText("Features: " + (", ".join(str(v) for v in feature_list) if feature_list else "-"))
        self._render_stage_summary_table()
        if self._processing_stages:
            self.stage_summary_table.selectRow(0)

    def _render_stage_summary_table(self) -> None:
        self.stage_summary_table.clear()
        headers = ["Stage", "Rows", "Missing", "In-spec", "Targets"]
        self.stage_summary_table.setColumnCount(len(headers))
        self.stage_summary_table.setRowCount(len(self._processing_stages))
        for ci, h in enumerate(headers):
            self.stage_summary_table.setHorizontalHeaderItem(ci, QTableWidgetItem(h))
        for ri, stage in enumerate(self._processing_stages):
            summary = stage.get("summary", {})
            self.stage_summary_table.setItem(ri, 0, QTableWidgetItem(str(stage.get("name", ""))))
            self.stage_summary_table.setItem(ri, 1, QTableWidgetItem(str(summary.get("row_count", 0))))
            self.stage_summary_table.setItem(ri, 2, QTableWidgetItem(str(summary.get("missing_values", 0))))
            self.stage_summary_table.setItem(ri, 3, QTableWidgetItem(str(summary.get("in_spec_count", 0))))
            self.stage_summary_table.setItem(ri, 4, QTableWidgetItem(str(summary.get("target_instance_count", 0))))
        vh = self.stage_summary_table.verticalHeader()
        vh.setDefaultSectionSize(24)
        vh.setMinimumSectionSize(20)

    def _on_stage_selection_changed(self) -> None:
        selected = self.stage_summary_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self._processing_stages):
            return
        stage = self._processing_stages[row]
        self._render_stage_preview(stage)

    def _render_stage_preview(self, stage: dict[str, Any]) -> None:
        name = str(stage.get("name", "-"))
        summary = stage.get("summary", {})
        self.stage_preview_title.setText(
            f"Preview: {name} | rows={summary.get('row_count', 0)} | missing={summary.get('missing_values', 0)}"
        )
        df = stage.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.stage_preview_table.setRowCount(0)
            self.stage_preview_table.setColumnCount(0)
            return
        show = df.head(40)
        cols = list(show.columns)
        self.stage_preview_table.clear()
        self.stage_preview_table.setRowCount(len(show))
        self.stage_preview_table.setColumnCount(len(cols))
        for ci, col_name in enumerate(cols):
            self.stage_preview_table.setHorizontalHeaderItem(ci, QTableWidgetItem(str(col_name)))
        for ri in range(len(show)):
            for ci, col_name in enumerate(cols):
                v = show.iloc[ri][col_name]
                txt = "" if pd.isna(v) else str(v)
                self.stage_preview_table.setItem(ri, ci, QTableWidgetItem(txt))
        vh = self.stage_preview_table.verticalHeader()
        vh.setDefaultSectionSize(22)
        vh.setMinimumSectionSize(20)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

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
        min_v = float(lifetimes.min())
        max_v = float(lifetimes.max())
        self.cutoff_spin.blockSignals(True)
        self.cutoff_spin.setMinimum(min_v)
        self.cutoff_spin.setMaximum(max_v)
        if target_id in self._cutoff_by_target:
            self.cutoff_spin.setValue(float(min(max(self._cutoff_by_target[target_id], min_v), max_v)))
        else:
            default_v = float(lifetimes.quantile(0.5))
            self.cutoff_spin.setValue(default_v)
            self._cutoff_by_target[target_id] = default_v
        self.cutoff_spin.blockSignals(False)

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

    def _refresh_benchmark_active_target_combo(
        self, *, active_target_id: str, target_ids: list[str]
    ) -> None:
        if self._material_dataset is None:
            return
        if not hasattr(self, "bench_active_target_combo"):
            return
        self.bench_active_target_combo.blockSignals(True)
        self.bench_active_target_combo.clear()
        for tid in target_ids:
            self.bench_active_target_combo.addItem(tid)
        if active_target_id and active_target_id in target_ids:
            self.bench_active_target_combo.setCurrentText(active_target_id)
        self.bench_active_target_combo.blockSignals(False)

        # Reuse cutoff entered on the Scenario Evaluation panel for consistency.
        if active_target_id and active_target_id in self._cutoff_by_target:
            self.bench_cutoff_spin.setValue(float(self._cutoff_by_target[active_target_id]))

    def _on_benchmark_split_mode_changed(self, split_mode: str) -> None:
        is_active_cutoff = str(split_mode).strip() == "active_target_cutoff"
        self.bench_active_cutoff_box.setVisible(is_active_cutoff)

    def _benchmark_run_clicked(self) -> None:
        if self._material_dataset is None:
            self.bench_status_label.setText("No material selected.")
            return

        df_all = self._material_dataset.all_records
        if df_all is None or df_all.empty:
            self.bench_status_label.setText("No rows available for benchmarking.")
            return

        selected_targets = self._selected_targets_from_widget(self.training_targets_list)
        if selected_targets:
            df = df_all[df_all["target_id"].astype(str).isin(selected_targets)].copy()
        else:
            df = df_all

        if df is None or df.empty:
            self.bench_status_label.setText("No rows available after target filtering.")
            return

        selected_bundles = self._selected_targets_from_widget(self.bench_bundle_list)
        if not selected_bundles:
            self.bench_status_label.setText("Select at least one model bundle.")
            return

        split_mode = self.bench_split_mode_combo.currentText().strip()

        # Assemble a benchmark config without mutating deployment config.
        bench_config = dict(self._config)
        bench_config["spec_config"] = self._material_dataset.spec_config
        bs = dict(bench_config.get("benchmark_settings") or {})

        # Ensure chosen mode has at least reasonable defaults.
        if split_mode == "forward_chaining":
            bs.setdefault("forward_chaining", {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4})
        elif split_mode == "leave_one_target_out":
            bs.setdefault("leave_one_target_out", {"min_train_rows": 10, "min_test_rows": 3})
        elif split_mode == "active_target_cutoff":
            bs.setdefault("active_target_cutoff", {})

            active_target_id = self.bench_active_target_combo.currentText().strip()
            cutoff_lifetime = float(self.bench_cutoff_spin.value())
            bs["active_target_cutoff"] = {
                **dict(bs.get("active_target_cutoff") or {}),
                "active_target_id": active_target_id,
                "cutoff_lifetime": cutoff_lifetime,
                # Benchmark runner supports future selection of history targets; keep empty for now.
                "history_target_ids": tuple(bs.get("active_target_cutoff", {}).get("history_target_ids") or ()),
            }

        bench_config["benchmark_settings"] = bs

        model_subset = {bn: MODEL_BUNDLE_PRESETS[bn] for bn in selected_bundles if bn in MODEL_BUNDLE_PRESETS}
        if not model_subset:
            self.bench_status_label.setText("No valid model bundles selected.")
            return

        self.bench_status_label.setText("Running benchmark...")
        self.bench_run_button.setEnabled(False)
        try:
            suite = run_prediction_benchmark(
                df,
                config=bench_config,
                model_names_by_target=model_subset,
                split_modes=[split_mode],
            )
        except Exception as exc:
            self.bench_status_label.setText("Benchmark failed.")
            self.bench_best_explanation.setPlainText(f"Benchmark error: {exc}")
            self.bench_run_button.setEnabled(True)
            return
        self.bench_run_button.setEnabled(True)

        self._last_benchmark_suite_results = suite
        ranked = rank_benchmark_results(suite, primary_metric="spec_pass_accuracy", secondary_metric="rs_mae")
        self._last_benchmark_ranked = ranked
        self._last_benchmark_best_bundle = ranked[0]["bundle_name"] if ranked else None

        self._render_benchmark_summary_table(suite)
        self._render_benchmark_fold_table(suite)
        self._render_benchmark_chart(suite)
        self._render_benchmark_best_explanation(suite, ranked)

        self.bench_status_label.setText("Completed.")

    def _benchmark_export_summary_csv(self) -> None:
        if self._last_benchmark_suite_results is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save benchmark summary CSV", "benchmark_summary.csv", "CSV Files (*.csv)"
        )
        if not out_path:
            return
        rows = self._last_benchmark_suite_results.to_table_rows()
        pd.DataFrame(rows).to_csv(out_path, index=False)

    def _benchmark_export_folds_csv(self) -> None:
        if self._last_benchmark_suite_results is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save benchmark folds CSV", "benchmark_folds.csv", "CSV Files (*.csv)"
        )
        if not out_path:
            return
        rows = flatten_benchmark_suite_folds(self._last_benchmark_suite_results)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    def _benchmark_adopt_winner_clicked(self) -> None:
        if self._last_benchmark_suite_results is None or not self._last_benchmark_best_bundle:
            self.bench_status_label.setText("Run benchmark first to adopt a winner.")
            return

        winner = self._last_benchmark_best_bundle
        preset = MODEL_BUNDLE_PRESETS.get(winner)
        if not preset:
            self.bench_status_label.setText("Winner preset not found.")
            return

        ms = dict(self._config.get("model_settings") or {})
        ms["rs_model"] = preset["rs"]
        ms["thickness_model"] = preset["thickness"]
        ms["rsu_model"] = preset["rsu"]
        self._config["model_settings"] = ms

        self.bench_status_label.setText(f"Winner adopted: {winner}. Click Train to retrain.")

    def _render_benchmark_summary_table(self, suite: Any) -> None:
        rows = suite.to_table_rows()
        headers = [
            "bundle_name",
            "split_mode",
            "rs_mae_mean",
            "thickness_mae_mean",
            "rsu_mae_mean",
            "spec_pass_accuracy_mean",
            "fold_count",
            "warnings",
        ]

        self.bench_summary_table.clear()
        self.bench_summary_table.setRowCount(len(rows))
        self.bench_summary_table.setColumnCount(len(headers))
        for ci, h in enumerate(headers):
            self.bench_summary_table.setHorizontalHeaderItem(ci, QTableWidgetItem(h))

        for ri, row in enumerate(rows):
            for ci, h in enumerate(headers):
                val = row.get(h)
                if isinstance(val, list):
                    sval = ";".join(str(x) for x in val)
                elif val is None:
                    sval = "-"
                else:
                    sval = f"{float(val):.6g}" if isinstance(val, (int, float)) else str(val)
                self.bench_summary_table.setItem(ri, ci, QTableWidgetItem(sval))

        self.bench_summary_table.resizeRowsToContents()

    def _render_benchmark_fold_table(self, suite: Any) -> None:
        rows = flatten_benchmark_suite_folds(suite)
        headers = [
            "bundle_name",
            "split_mode",
            "split_name",
            "train_row_count",
            "test_row_count",
            "rs_mae",
            "thickness_mae",
            "rsu_mae",
            "spec_pass_accuracy",
            "warnings",
        ]

        self.bench_fold_table.clear()
        self.bench_fold_table.setRowCount(len(rows))
        self.bench_fold_table.setColumnCount(len(headers))
        for ci, h in enumerate(headers):
            self.bench_fold_table.setHorizontalHeaderItem(ci, QTableWidgetItem(h))

        def fmt(v: Any) -> str:
            if v is None:
                return "-"
            if isinstance(v, (int, float)):
                return f"{float(v):.6g}"
            return str(v)

        for ri, row in enumerate(rows):
            for ci, h in enumerate(headers):
                self.bench_fold_table.setItem(ri, ci, QTableWidgetItem(fmt(row.get(h))))

        self.bench_fold_table.resizeRowsToContents()

    def _render_benchmark_chart(self, suite: Any) -> None:
        if self.bench_fig is None or self.bench_ax is None:
            return
        metric_label = self.bench_chart_metric_combo.currentText()
        metric_key_map = {
            "RS MAE": "rs_mae_mean",
            "Thickness MAE": "thickness_mae_mean",
            "RSU MAE": "rsu_mae_mean",
            "Spec pass accuracy": "spec_pass_accuracy_mean",
        }
        metric_key = metric_key_map.get(metric_label, "spec_pass_accuracy_mean")

        rows = suite.to_table_rows()
        labels = [r.get("bundle_name", "") for r in rows]
        values: list[float] = []
        for r in rows:
            v = r.get(metric_key)
            values.append(float(v) if v is not None else float("nan"))

        self.bench_ax.clear()
        x = list(range(len(labels)))
        self.bench_ax.bar(x, values, color="#4c78a8")
        self.bench_ax.set_xticks(x)
        self.bench_ax.set_xticklabels(labels, rotation=25, ha="right")
        self.bench_ax.set_title(f"Benchmark comparison: {metric_label}")
        self.bench_ax.grid(True, alpha=0.2, axis="y")
        self.bench_fig.tight_layout()
        if self.bench_canvas is not None:
            self.bench_canvas.draw_idle()

    def _render_benchmark_best_explanation(self, suite: Any, ranked: list[dict[str, Any]]) -> None:
        if not ranked:
            self.bench_best_explanation.setPlainText("No benchmark results to explain.")
            return

        best_bundle = ranked[0]["bundle_name"]
        best_run = None
        second_bundle = ranked[1]["bundle_name"] if len(ranked) > 1 else None
        second_run = None
        for run in suite.runs:
            if run.bundle_name == best_bundle:
                best_run = run
            if second_bundle and run.bundle_name == second_bundle:
                second_run = run

        if best_run is None:
            self.bench_best_explanation.setPlainText("Best bundle not found in suite runs.")
            return

        primary = ranked[0].get("primary_mean")
        secondary = ranked[0].get("secondary_mean")
        lines = [
            f"Best model bundle: {best_bundle}",
            f"Primary metric (spec-pass accuracy mean): {primary if primary is not None else '-'}",
            f"Secondary metric (RS MAE mean): {secondary if secondary is not None else '-'}",
            f"Fold count: {len(best_run.summary.folds)}",
        ]

        # Tradeoff vs second best (same primary metric ordering).
        if second_run is not None and second_bundle is not None:
            sec_primary = ranked[1].get("primary_mean")
            sec_secondary = ranked[1].get("secondary_mean")
            lines.append("")
            lines.append(f"Runner-up: {second_bundle}")
            lines.append(
                f"Runner-up primary={sec_primary if sec_primary is not None else '-'}, secondary(RS MAE)={sec_secondary if sec_secondary is not None else '-'}"
            )

        # Confidence heuristic: small test sets lower trust.
        if best_run.summary.folds:
            avg_test = sum(f.test_row_count for f in best_run.summary.folds) / max(len(best_run.summary.folds), 1)
            if avg_test < 5:
                lines.append("")
                lines.append(f"Confidence may be low: avg test rows per fold = {avg_test:.2f} (small holdout).")

        all_warnings: list[str] = []
        for f in best_run.summary.folds:
            if f.warnings:
                all_warnings.extend(f.warnings)
        if best_run.summary.aggregate_warnings:
            all_warnings.extend(best_run.summary.aggregate_warnings)

        if all_warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend([f"- {w}" for w in sorted(set(all_warnings))])

        self.bench_best_explanation.setPlainText("\n".join(lines))

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

