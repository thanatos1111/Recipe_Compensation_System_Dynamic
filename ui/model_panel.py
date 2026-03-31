"""
Model training and evaluation panel.
"""

from __future__ import annotations

from typing import Any, Optional
import threading

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
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
    QProgressBar,
    QScrollArea,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
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
from core.benchmark_chart_helpers import (
    BENCH_CHART_TAB_BACKTEST,
    BENCH_CHART_TAB_RSU,
    BENCH_CHART_TAB_RS,
    BENCH_CHART_TAB_SPEC_PASS,
    BENCH_CHART_TAB_THICKNESS,
    BENCH_CHART_TAB_UNCERTAINTY,
    backtest_chart_placeholder_text,
    benchmark_rows_have_any_uncertainty_metrics,
    plot_recommendation_backtest_rates,
    recommendation_backtest_has_chart_data,
    uncertainty_chart_placeholder_text,
)
from core.benchmark_ranking_display import format_uncertainty_ranking_explanation_lines
from core.benchmarking import (
    MODEL_BUNDLE_PRESETS,
    flatten_benchmark_suite_folds,
    run_prediction_benchmark,
)
from core.bundles import describe_bundle, get_bundle_catalog
from core.ranking import get_supported_ranking_objectives, get_supported_uncertainty_modes, rank_benchmark_suite
from core.response_models import train_material_models
from core.model_registry import get_model_availability_summary
from core.schemas import MaterialDataset

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    FigureCanvas = None  # type: ignore[assignment]

from matplotlib.figure import Figure


class _NoPropagateWheelListWidget(QListWidget):
    """
    QListWidget that accepts wheel events to avoid scrolling parent containers.

    In scrollable layouts (e.g. QScrollArea), wheel events can be re-routed to the
    parent even when the cursor is over the list, making both the list and the
    page scroll. Accepting the event here keeps scrolling localized.
    """

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        # Manually scroll the list and accept the event so the parent scroll area
        # doesn't also scroll.
        sb = self.verticalScrollBar()
        if sb is None:
            try:
                event.accept()
            except Exception:
                pass
            return

        delta_y = 0
        try:
            delta_y = int(event.angleDelta().y())
        except Exception:
            delta_y = 0

        # Typical mouse wheels use 120 units per "step".
        steps = 0
        if delta_y:
            steps = int(delta_y / 120) if abs(delta_y) >= 120 else (1 if delta_y > 0 else -1)

        if steps:
            single = max(int(sb.singleStep()), 1)
            sb.setValue(sb.value() - steps * single)

        try:
            event.accept()
        except Exception:
            pass
        # Do not call super().wheelEvent(event) (it can propagate at edges).


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
        self._last_benchmark_uncertainty_enabled: bool = False
        self._last_benchmark_ranking_objective: str = "spec_pass_first"
        self._last_benchmark_ranking_weights: dict[str, float] = {}
        self._last_benchmark_uncertainty_ranking_mode: str = "ignore"
        self._last_benchmark_uncertainty_ranking_weights: dict[str, float] = {}
        self._last_recommendation_backtest_result: Optional[Any] = None
        self._backtest_thread: Optional[QThread] = None
        self._backtest_worker: Optional[_BacktestWorker] = None
        self._backtest_abort_event: Optional[threading.Event] = None

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

        self.bench_bundle_list = _NoPropagateWheelListWidget()
        self.bench_bundle_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for bundle_name in MODEL_BUNDLE_PRESETS.keys():
            self.bench_bundle_list.addItem(QListWidgetItem(bundle_name))
        self.bench_bundle_list.setMinimumHeight(110)
        self.bench_bundle_list.itemSelectionChanged.connect(self._refresh_benchmark_bundle_details)

        # Prompt 10 + layout polish: keep bundles and bundle details aligned as left/right panels.
        # Left: multi-select bundle list. Right: bundle details.
        bundles_left_layout = QVBoxLayout()
        bundles_left_layout.addWidget(QLabel("Model bundles (multi-select)"))
        bundles_left_layout.addWidget(self.bench_bundle_list)
        bundles_left_layout.setContentsMargins(0, 0, 0, 0)
        bundles_left = QWidget()
        bundles_left.setLayout(bundles_left_layout)

        # Right panel: bundle details (read-only transparency for presets/custom bundles).
        self.bench_bundle_details_group = QGroupBox("Bundle details")
        bundle_details_layout = QVBoxLayout()
        self.bench_bundle_details_text = QTextEdit()
        self.bench_bundle_details_text.setReadOnly(True)
        self.bench_bundle_details_text.setMinimumHeight(110)
        self.bench_bundle_details_text.setPlaceholderText("Select a bundle to see what it contains.")
        bundle_details_layout.addWidget(self.bench_bundle_details_text)
        self.bench_bundle_details_group.setLayout(bundle_details_layout)

        bundles_split = QSplitter()
        bundles_split.setOrientation(Qt.Orientation.Horizontal)
        bundles_split.addWidget(bundles_left)
        bundles_split.addWidget(self.bench_bundle_details_group)
        bundles_split.setStretchFactor(0, 0)
        bundles_split.setStretchFactor(1, 1)
        bundles_split.setCollapsible(0, False)
        bundles_split.setCollapsible(1, False)
        bundles_split.setSizes([220, 520])

        benchmark_form.addRow("Bundles:", bundles_split)

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

        # Ranking objective controls (Prompt 8.1).
        self.bench_ranking_objective_combo = QComboBox()
        self.bench_ranking_objective_combo.addItems(
            ["spec_pass_first", "rs_first", "thickness_first", "rsu_first", "weighted_combined"]
        )
        self.bench_ranking_objective_combo.setCurrentText("spec_pass_first")
        benchmark_form.addRow("Ranking objective:", self.bench_ranking_objective_combo)

        self.bench_weighted_config_box = QGroupBox("Weighted score configuration")
        weighted_form = QFormLayout()
        weighted_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        def _make_weight_spin(default: float) -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1000.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setKeyboardTracking(False)
            sp.setValue(float(default))
            return sp

        self.bench_weight_spec_pass_spin = _make_weight_spin(1.0)
        self.bench_weight_rs_mae_spin = _make_weight_spin(1.0)
        self.bench_weight_thickness_mae_spin = _make_weight_spin(1.0)
        self.bench_weight_rsu_mae_spin = _make_weight_spin(1.0)

        weighted_form.addRow("spec_pass_accuracy weight:", self.bench_weight_spec_pass_spin)
        weighted_form.addRow("rs_mae weight:", self.bench_weight_rs_mae_spin)
        weighted_form.addRow("thickness_mae weight:", self.bench_weight_thickness_mae_spin)
        weighted_form.addRow("rsu_mae weight:", self.bench_weight_rsu_mae_spin)
        self.bench_weighted_config_box.setLayout(weighted_form)
        benchmark_form.addRow(self.bench_weighted_config_box)

        def _on_objective_changed(obj: str) -> None:
            self.bench_weighted_config_box.setVisible(obj.strip() == "weighted_combined")

        self.bench_ranking_objective_combo.currentTextChanged.connect(_on_objective_changed)
        _on_objective_changed(self.bench_ranking_objective_combo.currentText())

        # Uncertainty ranking mode (Prompt 9.0a): how interval quality affects winner selection.
        self.bench_uncertainty_ranking_mode_combo = QComboBox()
        for mode in ("ignore", "warn_only", "include_in_score"):
            self.bench_uncertainty_ranking_mode_combo.addItem(mode)
        self.bench_uncertainty_ranking_mode_combo.setCurrentText("ignore")
        um = get_supported_uncertainty_modes()
        for i in range(self.bench_uncertainty_ranking_mode_combo.count()):
            key = self.bench_uncertainty_ranking_mode_combo.itemText(i)
            tip = str(um.get(key, {}).get("description") or "").strip()
            if tip:
                self.bench_uncertainty_ranking_mode_combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        help_lbl = QLabel(
            "Rank by metrics only; warn without changing rank; or blend interval quality into ranking."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: palette(mid);")
        unc_rank_row = QVBoxLayout()
        unc_rank_row.addWidget(self.bench_uncertainty_ranking_mode_combo)
        unc_rank_row.addWidget(help_lbl)
        unc_rank_wrap = QWidget()
        unc_rank_wrap.setLayout(unc_rank_row)
        benchmark_form.addRow("Uncertainty ranking mode:", unc_rank_wrap)

        self.bench_interval_quality_weight_spin = QDoubleSpinBox()
        self.bench_interval_quality_weight_spin.setRange(0.0, 1000.0)
        self.bench_interval_quality_weight_spin.setDecimals(3)
        self.bench_interval_quality_weight_spin.setSingleStep(0.1)
        self.bench_interval_quality_weight_spin.setKeyboardTracking(False)
        self.bench_interval_quality_weight_spin.setValue(1.0)

        def _on_uncertainty_ranking_mode_changed(mode: str) -> None:
            show_w = mode.strip() == "include_in_score"
            self.bench_interval_quality_weight_spin.setVisible(show_w)
            self.bench_interval_quality_weight_spin_label.setVisible(show_w)

        self.bench_interval_quality_weight_spin_label = QLabel("interval_quality weight (weighted combined):")
        self.bench_interval_quality_weight_spin_label.setVisible(False)
        self.bench_interval_quality_weight_spin.setVisible(False)
        self.bench_uncertainty_ranking_mode_combo.currentTextChanged.connect(_on_uncertainty_ranking_mode_changed)
        _on_uncertainty_ranking_mode_changed(self.bench_uncertainty_ranking_mode_combo.currentText())
        benchmark_form.addRow(self.bench_interval_quality_weight_spin_label, self.bench_interval_quality_weight_spin)

        # Optional uncertainty configuration (conformal interval quality).
        self.bench_uncertainty_enabled_checkbox = QCheckBox("Enable uncertainty (conformal intervals)")
        self.bench_uncertainty_enabled_checkbox.setChecked(False)

        self.bench_uncertainty_alpha_spin = QDoubleSpinBox()
        self.bench_uncertainty_alpha_spin.setRange(0.01, 0.5)
        self.bench_uncertainty_alpha_spin.setDecimals(3)
        self.bench_uncertainty_alpha_spin.setKeyboardTracking(False)
        self.bench_uncertainty_alpha_spin.setValue(0.1)

        self.bench_uncertainty_calib_frac_spin = QDoubleSpinBox()
        self.bench_uncertainty_calib_frac_spin.setRange(0.05, 0.8)
        self.bench_uncertainty_calib_frac_spin.setDecimals(3)
        self.bench_uncertainty_calib_frac_spin.setKeyboardTracking(False)
        self.bench_uncertainty_calib_frac_spin.setValue(0.2)

        def _on_uncertainty_toggled(checked: bool) -> None:
            self.bench_uncertainty_alpha_spin.setEnabled(checked)
            self.bench_uncertainty_calib_frac_spin.setEnabled(checked)

        self.bench_uncertainty_enabled_checkbox.toggled.connect(_on_uncertainty_toggled)
        _on_uncertainty_toggled(False)

        benchmark_form.addRow(self.bench_uncertainty_enabled_checkbox)
        benchmark_form.addRow("Conformal alpha:", self.bench_uncertainty_alpha_spin)
        benchmark_form.addRow("Calibration fraction:", self.bench_uncertainty_calib_frac_spin)

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

        # Model availability summary (external boosted models are optional).
        benchmark_availability_group = QGroupBox("Model availability")
        availability_layout = QVBoxLayout()
        self.bench_model_availability_label = QLabel("")
        self.bench_model_availability_label.setWordWrap(True)
        availability_layout.addWidget(self.bench_model_availability_label)
        benchmark_availability_group.setLayout(availability_layout)
        benchmark_layout.addWidget(benchmark_availability_group)
        self._refresh_benchmark_model_availability()

        self.bench_summary_table = QTableWidget()
        # Ensure full header text is visible.
        # If total column width exceeds available space, QTableWidget will
        # show a horizontal scrollbar (see policy below).
        self.bench_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.bench_summary_table.horizontalHeader().setStretchLastSection(False)
        self.bench_summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.bench_summary_table.setMinimumHeight(120)
        benchmark_layout.addWidget(self.bench_summary_table)

        self.bench_fold_table = QTableWidget()
        # Same approach as bench_summary_table.
        self.bench_fold_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.bench_fold_table.horizontalHeader().setStretchLastSection(False)
        self.bench_fold_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.bench_fold_table.setMinimumHeight(180)
        benchmark_layout.addWidget(self.bench_fold_table)

        # Recommendation backtesting subsection (Milestone 6).
        self.bench_backtest_group = QGroupBox("Recommendation Backtest (fold replay)")
        backtest_layout = QVBoxLayout()

        self.bench_backtest_enabled_checkbox = QCheckBox("Enable recommendation backtest after benchmark")
        self.bench_backtest_enabled_checkbox.setChecked(False)
        self.bench_backtest_enabled_checkbox.toggled.connect(self._on_bench_backtest_toggled)
        backtest_layout.addWidget(self.bench_backtest_enabled_checkbox)

        # Progress + controls (must stay responsive via QThread).
        progress_row = QHBoxLayout()
        self.bench_backtest_progress = QProgressBar()
        self.bench_backtest_progress.setRange(0, 100)
        self.bench_backtest_progress.setValue(0)
        self.bench_backtest_progress.setTextVisible(True)
        self.bench_backtest_progress.setFormat("%p%")
        progress_row.addWidget(self.bench_backtest_progress, 1)

        self.bench_backtest_max_test_rows_spin = QSpinBox()
        self.bench_backtest_max_test_rows_spin.setRange(1, 100000)
        self.bench_backtest_max_test_rows_spin.setValue(50)
        progress_row.addWidget(QLabel("Max test rows/fold:"))
        progress_row.addWidget(self.bench_backtest_max_test_rows_spin)

        self.bench_backtest_abort_button = QPushButton("Abort backtest")
        self.bench_backtest_abort_button.setEnabled(False)
        self.bench_backtest_abort_button.clicked.connect(self._on_bench_backtest_abort_clicked)
        progress_row.addWidget(self.bench_backtest_abort_button)

        backtest_layout.addLayout(progress_row)

        self.bench_backtest_table = QTableWidget()
        # Keep long column titles visible; show horizontal scroll when needed.
        self.bench_backtest_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.bench_backtest_table.horizontalHeader().setStretchLastSection(False)
        self.bench_backtest_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.bench_backtest_table.setMinimumHeight(150)
        backtest_layout.addWidget(self.bench_backtest_table)

        self.bench_backtest_fig = Figure(figsize=(6, 3.2))
        self.bench_backtest_ax = self.bench_backtest_fig.add_subplot(111)
        if FigureCanvas is not None:
            self.bench_backtest_canvas = FigureCanvas(self.bench_backtest_fig)
            self.bench_backtest_canvas.setMinimumHeight(160)
            backtest_layout.addWidget(self.bench_backtest_canvas)
        else:
            self.bench_backtest_canvas = None
            backtest_layout.addWidget(QLabel("Matplotlib Qt canvas not available."))

        self.bench_backtest_group.setLayout(backtest_layout)
        # Keep the checkbox + progress controls visible; only the table/plot
        # are toggled when the user enables/disables the backtest.
        self.bench_backtest_group.setVisible(True)
        benchmark_layout.addWidget(self.bench_backtest_group)

        # Backtest warnings summary (compact, deduped).
        self.bench_backtest_warning_box = QGroupBox("Backtest warnings")
        backtest_warn_layout = QVBoxLayout()
        self.bench_backtest_warnings_text = QTextEdit()
        self.bench_backtest_warnings_text.setReadOnly(True)
        self.bench_backtest_warnings_text.setMaximumHeight(90)
        backtest_warn_layout.addWidget(self.bench_backtest_warnings_text)
        self.bench_backtest_warning_box.setLayout(backtest_warn_layout)
        self.bench_backtest_warning_box.setVisible(False)
        benchmark_layout.addWidget(self.bench_backtest_warning_box)

        bench_bottom_split = QSplitter()
        bench_bottom_split.setOrientation(Qt.Orientation.Horizontal)

        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        self.bench_fig = Figure(figsize=(6, 4))
        self.bench_chart_tab_bar = QTabBar()
        for label in ("Spec-pass", "RS", "Thickness", "RSU", "Uncertainty", "Backtest"):
            self.bench_chart_tab_bar.addTab(label)
        self.bench_chart_tab_bar.currentChanged.connect(self._on_bench_chart_tab_changed)
        chart_layout.addWidget(self.bench_chart_tab_bar)
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

        # Benchmark warnings summary (compact, deduped).
        self.bench_warning_box = QGroupBox("Benchmark warnings")
        warn_layout = QVBoxLayout()
        self.bench_warnings_text = QTextEdit()
        self.bench_warnings_text.setReadOnly(True)
        self.bench_warnings_text.setMaximumHeight(90)
        warn_layout.addWidget(self.bench_warnings_text)
        self.bench_warning_box.setLayout(warn_layout)
        self.bench_warning_box.setVisible(False)
        benchmark_layout.addWidget(self.bench_warning_box)
        benchmark_page.setLayout(benchmark_page_layout)
        self.section_tabs.addTab(benchmark_page, "5) Benchmark")
        self._on_benchmark_split_mode_changed(self.bench_split_mode_combo.currentText())

        # Initialize backtest widgets to the "disabled" state.
        self._on_bench_backtest_toggled(bool(self.bench_backtest_enabled_checkbox.isChecked()))

        self._refresh_bench_chart()

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

    def _refresh_benchmark_model_availability(self) -> None:
        if not hasattr(self, "bench_model_availability_label"):
            return

        summary = get_model_availability_summary()

        # sklearn models are always available in this app.
        lines: list[str] = ["sklearn models: available"]
        for mn in ("xgb", "lgbm", "catboost"):
            meta = summary.get(mn) or {}
            available = bool(meta.get("available"))
            if available:
                lines.append(f"{mn}: available")
                continue

            missing_error = meta.get("missing_error") or ""
            install = missing_error.strip() if isinstance(missing_error, str) else ""
            if not install:
                dep = meta.get("external_dependency") or mn
                install = f"pip install {dep}"
            lines.append(f"{mn}: missing (install: {install})")

        self.bench_model_availability_label.setText("\n".join(lines))

    def _refresh_benchmark_bundle_details(self) -> None:
        if not hasattr(self, "bench_bundle_details_text") or not hasattr(self, "bench_bundle_list"):
            return

        selected = self._selected_targets_from_widget(self.bench_bundle_list)
        if not selected:
            self.bench_bundle_details_text.setPlainText("Select a bundle to see what it contains.")
            return

        # Requirement: keep multi-select; show first selected details (compact).
        # (Optionally we could show a compact list for multi-select, but we keep
        # this minimal for Prompt 10.)
        bn = str(selected[0])
        try:
            desc = describe_bundle(bn, catalog=get_bundle_catalog())
        except Exception as exc:
            self.bench_bundle_details_text.setPlainText(f"Unable to describe bundle {bn!r}: {exc}")
            return

        by_target = desc.model_by_target()

        def _line(target: str, label: str) -> str:
            m = by_target.get(target)
            if m is None:
                return f"- {label}: (missing)"
            if m.display_name and m.display_name != m.model_name:
                return f"- {label}: {m.model_name} ({m.display_name})"
            return f"- {label}: {m.model_name}"

        lines = [
            f"Bundle: {desc.bundle_name}",
            f"Source: {desc.source}",
            "",
            _line("rs", "RS"),
            _line("thickness", "Thickness"),
            _line("rsu", "RSU"),
        ]
        if len(selected) > 1:
            more = ", ".join(str(x) for x in selected[1:])
            lines += ["", f"Also selected: {more}"]

        self.bench_bundle_details_text.setPlainText("\n".join(lines))

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _render_benchmark_warning_summary(self, suite: Any) -> None:
        if not hasattr(self, "bench_warning_box") or not hasattr(self, "bench_warnings_text"):
            return
        if suite is None or not getattr(suite, "runs", None):
            self.bench_warning_box.setVisible(False)
            self.bench_warnings_text.setPlainText("")
            return

        lines: list[str] = []
        for run in suite.runs:
            try:
                warns = list(run.summary.aggregate_warnings or [])
            except Exception:
                warns = []
            for w in warns:
                if not w:
                    continue
                lines.append(f"{run.bundle_name}: {str(w)}")

        lines = self._dedupe_preserve_order([str(x) for x in lines if str(x).strip()])
        if not lines:
            self.bench_warning_box.setVisible(False)
            self.bench_warnings_text.setPlainText("")
            return

        self.bench_warning_box.setVisible(True)
        self.bench_warnings_text.setPlainText("\n".join(lines))

    def _render_backtest_warning_summary(self, backtest_result: Any) -> None:
        if not hasattr(self, "bench_backtest_warning_box") or not hasattr(self, "bench_backtest_warnings_text"):
            return
        if backtest_result is None:
            self.bench_backtest_warning_box.setVisible(False)
            self.bench_backtest_warnings_text.setPlainText("")
            return

        try:
            rows = list(backtest_result.to_table_rows() or [])
        except Exception:
            rows = []

        lines: list[str] = []
        for r in rows:
            bn = str(r.get("bundle_name") or "")
            warn_str = str(r.get("warnings") or "").strip()
            rows_eval = int(r.get("rows_evaluated") or 0)
            if not warn_str:
                continue
            for w in [x.strip() for x in warn_str.split(";") if x.strip()]:
                suffix = " (no rows evaluated)" if rows_eval == 0 else ""
                lines.append(f"{bn}: {w}{suffix}")

        lines = self._dedupe_preserve_order([str(x) for x in lines if str(x).strip()])
        if not lines:
            self.bench_backtest_warning_box.setVisible(False)
            self.bench_backtest_warnings_text.setPlainText("")
            return

        self.bench_backtest_warning_box.setVisible(True)
        self.bench_backtest_warnings_text.setPlainText("\n".join(lines))

    def _on_bench_backtest_toggled(self, checked: bool) -> None:
        self._last_recommendation_backtest_result = None
        self.bench_backtest_progress.setValue(0)
        self.bench_backtest_abort_button.setEnabled(False)
        self.bench_backtest_max_test_rows_spin.setEnabled(bool(checked))
        self.bench_backtest_table.clear()
        self.bench_backtest_table.setRowCount(0)
        self.bench_backtest_table.setColumnCount(0)
        if self.bench_backtest_ax is not None:
            self.bench_backtest_ax.clear()
            self.bench_backtest_ax.grid(True, alpha=0.2)
            if self.bench_backtest_canvas is not None:
                self.bench_backtest_canvas.draw_idle()

    def _on_bench_backtest_abort_clicked(self) -> None:
        if self._backtest_abort_event is not None:
            self._backtest_abort_event.set()
            self.bench_status_label.setText("Aborting recommendation backtest...")
            self.bench_backtest_abort_button.setEnabled(False)

    def _on_backtest_progress(self, percent: int, stage: str) -> None:
        # Runs in the UI thread via Qt queued signal.
        try:
            self.bench_backtest_progress.setValue(int(percent))
        except Exception:
            return
        if stage:
            self.bench_status_label.setText(f"Recommendation backtest... {int(percent)}% ({stage})")
        else:
            self.bench_status_label.setText(f"Recommendation backtest... {int(percent)}%")

    def _render_recommendation_backtest(self, backtest_result: Any) -> None:
        if backtest_result is None:
            self._on_bench_backtest_toggled(False)
            return

        rows = backtest_result.to_table_rows()
        headers = [
            "bundle_name",
            "rows_evaluated",
            "recommendation_improvement_rate",
            "predicted_spec_pass_improvement_rate",
            "no_change_fraction",
            "move_mean_abs_total",
            "move_median_abs_total",
            "move_max_abs_total",
            "warnings",
        ]

        self.bench_backtest_table.clear()
        self.bench_backtest_table.setRowCount(len(rows))
        self.bench_backtest_table.setColumnCount(len(headers))
        for ci, h in enumerate(headers):
            self.bench_backtest_table.setHorizontalHeaderItem(ci, QTableWidgetItem(h))

        def fmt_val(v: Any, *, col: str) -> str:
            if v is None:
                return "-"
            if isinstance(v, (int, float)):
                if col.endswith("_rate") or col.endswith("fraction"):
                    return f"{float(v) * 100.0:.1f}%"
                return f"{float(v):.6g}"
            return str(v)

        for ri, row in enumerate(rows):
            for ci, h in enumerate(headers):
                self.bench_backtest_table.setItem(ri, ci, QTableWidgetItem(fmt_val(row.get(h), col=h)))

        self.bench_backtest_table.resizeRowsToContents()
        self.bench_backtest_table.resizeColumnsToContents()
        self._render_backtest_warning_summary(backtest_result)

        # Simple plots: compare rates side-by-side.
        if self.bench_backtest_ax is None:
            return

        self.bench_backtest_ax.clear()
        if not rows:
            self.bench_backtest_ax.set_title("No backtest data")
            if self.bench_backtest_canvas is not None:
                self.bench_backtest_canvas.draw_idle()
            return

        plot_recommendation_backtest_rates(self.bench_backtest_ax, rows)

        if self.bench_backtest_canvas is not None:
            self.bench_backtest_canvas.draw_idle()

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

        self._last_benchmark_uncertainty_enabled = bool(self.bench_uncertainty_enabled_checkbox.isChecked())
        if self._last_benchmark_uncertainty_enabled:
            bs["uncertainty"] = {
                "enabled": True,
                "alpha": float(self.bench_uncertainty_alpha_spin.value()),
                "calibration_fraction": float(self.bench_uncertainty_calib_frac_spin.value()),
                # Keep the minimum small so conformal intervals work in small folds.
                "min_calibration_rows": 5,
            }
        else:
            bs.pop("uncertainty", None)

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
        objective = self.bench_ranking_objective_combo.currentText().strip() or "spec_pass_first"
        weights: Optional[dict[str, float]] = None
        weight_warning = ""
        if objective == "weighted_combined":
            weights = {
                "spec_pass_accuracy": float(self.bench_weight_spec_pass_spin.value()),
                "rs_mae": float(self.bench_weight_rs_mae_spin.value()),
                "thickness_mae": float(self.bench_weight_thickness_mae_spin.value()),
                "rsu_mae": float(self.bench_weight_rsu_mae_spin.value()),
            }
            if all(abs(float(v)) <= 0.0 for v in weights.values()):
                weight_warning = (
                    "Weighted combined selected, but all weights are zero. "
                    "Ranking will be deterministic but may be uninformative."
                )

        uncertainty_mode = (self.bench_uncertainty_ranking_mode_combo.currentText().strip() or "ignore").lower()
        if uncertainty_mode not in {"ignore", "warn_only", "include_in_score"}:
            uncertainty_mode = "ignore"
        uncertainty_weights: Optional[dict[str, float]] = None
        if uncertainty_mode == "include_in_score":
            uncertainty_weights = {"interval_quality": float(self.bench_interval_quality_weight_spin.value())}

        conformal_alpha: Optional[float] = None
        if self._last_benchmark_uncertainty_enabled:
            conformal_alpha = float(self.bench_uncertainty_alpha_spin.value())

        try:
            ranked = rank_benchmark_suite(
                suite,
                objective=objective,
                weights=weights,
                uncertainty_mode=uncertainty_mode,
                uncertainty_weights=uncertainty_weights,
                conformal_alpha=conformal_alpha,
            )
        except Exception as exc:
            self.bench_status_label.setText("Ranking failed.")
            self.bench_best_explanation.setPlainText(f"Ranking error: {exc}")
            return

        self._last_benchmark_ranked = ranked
        self._last_benchmark_best_bundle = ranked[0]["bundle_name"] if ranked else None
        self._last_benchmark_ranking_objective = objective
        self._last_benchmark_ranking_weights = dict(weights or {})
        self._last_benchmark_uncertainty_ranking_mode = uncertainty_mode
        self._last_benchmark_uncertainty_ranking_weights = dict(uncertainty_weights or {})

        self._render_benchmark_summary_table(suite)
        self._render_benchmark_fold_table(suite)
        self._refresh_bench_chart()
        self._render_benchmark_best_explanation(suite, ranked)
        self._render_benchmark_warning_summary(suite)
        if weight_warning:
            self.bench_status_label.setText(weight_warning)

        # Optional recommendation backtest.
        if self.bench_backtest_enabled_checkbox.isChecked():
            self._start_backtest_worker(df, bench_config, selected_bundles, split_mode)
        else:
            self._on_bench_backtest_toggled(False)

        if self.bench_backtest_enabled_checkbox.isChecked():
            self.bench_status_label.setText("Benchmark completed. Waiting for recommendation backtest...")
        else:
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
        if self._last_benchmark_uncertainty_enabled:
            headers = [
                *headers[:],
                "rs_interval_coverage_mean",
                "rs_interval_mean_width_mean",
                "thickness_interval_coverage_mean",
                "thickness_interval_mean_width_mean",
                "rsu_interval_coverage_mean",
                "rsu_interval_mean_width_mean",
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
        # Update column widths based on (potentially longer) header text.
        self.bench_summary_table.resizeColumnsToContents()

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
        if self._last_benchmark_uncertainty_enabled:
            headers = [
                *headers[:9],
                "rs_interval_coverage",
                "rs_interval_mean_width",
                "thickness_interval_coverage",
                "thickness_interval_mean_width",
                "rsu_interval_coverage",
                "rsu_interval_mean_width",
                *headers[9:],
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
        # Update column widths based on (potentially longer) header text.
        self.bench_fold_table.resizeColumnsToContents()

    def _on_bench_chart_tab_changed(self, _index: int) -> None:
        self._refresh_bench_chart()

    def _render_benchmark_chart_placeholder(self, ax: Any, message: str) -> None:
        ax.clear()
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, wrap=True, transform=ax.transAxes)

    def _render_benchmark_chart_spec_pass(self, ax: Any, rows: list[dict[str, Any]]) -> None:
        labels = [r.get("bundle_name", "") for r in rows]
        values = [
            float(v) if (v := r.get("spec_pass_accuracy_mean")) is not None else float("nan") for r in rows
        ]
        x = list(range(len(labels)))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Benchmark: spec-pass accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

    def _render_benchmark_chart_rs(self, ax: Any, rows: list[dict[str, Any]]) -> None:
        labels = [r.get("bundle_name", "") for r in rows]
        values = [float(v) if (v := r.get("rs_mae_mean")) is not None else float("nan") for r in rows]
        x = list(range(len(labels)))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Benchmark: RS MAE (lower is better)")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.2, axis="y")

    def _render_benchmark_chart_thickness(self, ax: Any, rows: list[dict[str, Any]]) -> None:
        labels = [r.get("bundle_name", "") for r in rows]
        values = [float(v) if (v := r.get("thickness_mae_mean")) is not None else float("nan") for r in rows]
        x = list(range(len(labels)))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Benchmark: thickness MAE (lower is better)")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.2, axis="y")

    def _render_benchmark_chart_rsu(self, ax: Any, rows: list[dict[str, Any]]) -> None:
        labels = [r.get("bundle_name", "") for r in rows]
        values = [float(v) if (v := r.get("rsu_mae_mean")) is not None else float("nan") for r in rows]
        x = list(range(len(labels)))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Benchmark: RSU MAE (lower is better)")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.2, axis="y")

    def _render_benchmark_chart_uncertainty(self, rows: list[dict[str, Any]]) -> None:
        msg = uncertainty_chart_placeholder_text(
            uncertainty_enabled_for_run=bool(self._last_benchmark_uncertainty_enabled),
            metrics_available=benchmark_rows_have_any_uncertainty_metrics(rows),
        )
        if msg:
            ax = self.bench_fig.add_subplot(111)
            self._render_benchmark_chart_placeholder(ax, msg)
            return

        # Matplotlib versions vary in supported kwargs for Figure.subplots.
        # Keep this compatible with older installs (e.g. 3.1x).
        try:
            self.bench_fig.set_constrained_layout(True)
        except Exception:
            pass
        axes = self.bench_fig.subplots(2, 3)
        bundles = [str(r.get("bundle_name", "")) for r in rows]
        x = np.arange(len(bundles))
        nominal = 1.0 - float(self.bench_uncertainty_alpha_spin.value())

        cov_specs: tuple[tuple[str, str], ...] = (
            ("rs_interval_coverage_mean", "RS coverage"),
            ("thickness_interval_coverage_mean", "Thickness coverage"),
            ("rsu_interval_coverage_mean", "RSU coverage"),
        )
        for j, (key, title) in enumerate(cov_specs):
            ax_u = axes[0, j]
            vals = [float(r[key]) if r.get(key) is not None else float("nan") for r in rows]
            ax_u.bar(x, vals, color="#4c78a8")
            line_kw = {"label": "Nominal (1 − α)"} if j == 0 else {}
            ax_u.axhline(nominal, color="#c45a4a", linestyle="--", linewidth=1.0, **line_kw)
            ax_u.set_xticks(x)
            ax_u.set_xticklabels(bundles, rotation=25, ha="right", fontsize=8)
            ax_u.set_title(title, fontsize=9)
            ax_u.set_ylim(0.0, 1.05)
            ax_u.grid(True, alpha=0.2, axis="y")
            if j == 0:
                ax_u.legend(fontsize=7, loc="lower right")

        width_specs: tuple[tuple[str, str], ...] = (
            ("rs_interval_mean_width_mean", "RS mean interval width"),
            ("thickness_interval_mean_width_mean", "Thickness mean interval width"),
            ("rsu_interval_mean_width_mean", "RSU mean interval width"),
        )
        for j, (key, title) in enumerate(width_specs):
            ax_u = axes[1, j]
            vals = [float(r[key]) if r.get(key) is not None else float("nan") for r in rows]
            ax_u.bar(x, vals, color="#72b7b2")
            ax_u.set_xticks(x)
            ax_u.set_xticklabels(bundles, rotation=25, ha="right", fontsize=8)
            ax_u.set_title(title, fontsize=9)
            ax_u.grid(True, alpha=0.2, axis="y")

    def _render_benchmark_chart_backtest(self, ax: Any) -> None:
        bt = self._last_recommendation_backtest_result
        if not recommendation_backtest_has_chart_data(bt):
            self._render_benchmark_chart_placeholder(ax, backtest_chart_placeholder_text(False))
            return
        plot_recommendation_backtest_rates(ax, bt.to_table_rows())

    def _refresh_bench_chart(self) -> None:
        if self.bench_fig is None or self.bench_canvas is None:
            return
        tab = int(self.bench_chart_tab_bar.currentIndex())
        suite = self._last_benchmark_suite_results
        rows: list[dict[str, Any]] = list(suite.to_table_rows()) if suite is not None else []

        self.bench_fig.clf()

        if tab == BENCH_CHART_TAB_BACKTEST:
            ax = self.bench_fig.add_subplot(111)
            self._render_benchmark_chart_backtest(ax)
        elif tab == BENCH_CHART_TAB_UNCERTAINTY:
            self._render_benchmark_chart_uncertainty(rows)
        else:
            ax = self.bench_fig.add_subplot(111)
            if not rows:
                self._render_benchmark_chart_placeholder(
                    ax, "No benchmark results yet. Run a benchmark to compare bundles."
                )
            elif tab == BENCH_CHART_TAB_SPEC_PASS:
                self._render_benchmark_chart_spec_pass(ax, rows)
            elif tab == BENCH_CHART_TAB_RS:
                self._render_benchmark_chart_rs(ax, rows)
            elif tab == BENCH_CHART_TAB_THICKNESS:
                self._render_benchmark_chart_thickness(ax, rows)
            elif tab == BENCH_CHART_TAB_RSU:
                self._render_benchmark_chart_rsu(ax, rows)
            else:
                self._render_benchmark_chart_placeholder(ax, "Unknown chart selection.")

        if tab != BENCH_CHART_TAB_UNCERTAINTY:
            try:
                self.bench_fig.tight_layout()
            except Exception:
                pass
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

        obj = self._last_benchmark_ranking_objective or "spec_pass_first"
        meta = get_supported_ranking_objectives().get(obj, {})

        lines: list[str] = []
        lines.append(f"Ranking objective: {obj}")
        dn = str(meta.get("display_name") or "").strip()
        desc = str(meta.get("description") or "").strip()
        if dn:
            lines.append(f"- {dn}")
        if desc:
            lines.append(f"- {desc}")

        if obj == "weighted_combined":
            w = dict(self._last_benchmark_ranking_weights or {})
            lines.append("Weights:")
            lines.append(f"- spec_pass_accuracy: {w.get('spec_pass_accuracy', 0.0)}")
            lines.append(f"- rs_mae: {w.get('rs_mae', 0.0)}")
            lines.append(f"- thickness_mae: {w.get('thickness_mae', 0.0)}")
            lines.append(f"- rsu_mae: {w.get('rsu_mae', 0.0)}")
            if all(abs(float(v)) <= 0.0 for v in w.values()):
                lines.append("Warning: all weights are zero; combined ranking may be uninformative.")

        umode = self._last_benchmark_uncertainty_ranking_mode or "ignore"
        lines.append("")
        lines.extend(
            format_uncertainty_ranking_explanation_lines(
                ranked,
                uncertainty_mode=umode,
                uncertainty_weights=self._last_benchmark_uncertainty_ranking_weights or None,
                uncertainty_metrics_enabled=bool(self._last_benchmark_uncertainty_enabled),
                ranking_objective=obj,
            )
        )

        lines.append("")
        lines.append(f"Best model bundle: {best_bundle}")

        primary = ranked[0].get("primary_mean")
        secondary = ranked[0].get("secondary_mean")
        if obj == "spec_pass_first":
            lines.append(f"Primary metric (spec-pass accuracy mean): {primary if primary is not None else '-'}")
            lines.append(f"Secondary metric (RS MAE mean): {secondary if secondary is not None else '-'}")
        elif obj == "rs_first":
            lines.append(f"Primary metric (RS MAE mean): {primary if primary is not None else '-'}")
            lines.append(f"Secondary metric (spec-pass accuracy mean): {secondary if secondary is not None else '-'}")
        elif obj == "thickness_first":
            lines.append(f"Primary metric (Thickness MAE mean): {primary if primary is not None else '-'}")
            lines.append(f"Secondary metric (spec-pass accuracy mean): {secondary if secondary is not None else '-'}")
        elif obj == "rsu_first":
            lines.append(f"Primary metric (RSU MAE mean): {primary if primary is not None else '-'}")
            lines.append(f"Secondary metric (spec-pass accuracy mean): {secondary if secondary is not None else '-'}")
        elif obj == "weighted_combined":
            ws = ranked[0].get("weighted_score")
            lines.append(f"Weighted score: {ws if ws is not None else '-'}")

        lines.append(f"Fold count: {len(best_run.summary.folds)}")

        # Tradeoff vs second best (same primary metric ordering).
        if second_run is not None and second_bundle is not None:
            sec_primary = ranked[1].get("primary_mean")
            sec_secondary = ranked[1].get("secondary_mean")
            lines.append("")
            lines.append(f"Runner-up: {second_bundle}")
            if obj == "weighted_combined":
                sec_ws = ranked[1].get("weighted_score")
                lines.append(f"Runner-up weighted_score={sec_ws if sec_ws is not None else '-'}")
            else:
                lines.append(
                    f"Runner-up primary={sec_primary if sec_primary is not None else '-'}, secondary={sec_secondary if sec_secondary is not None else '-'}"
                )

        # Confidence heuristic: small test sets lower trust.
        if best_run.summary.folds:
            avg_test = sum(f.test_row_count for f in best_run.summary.folds) / max(len(best_run.summary.folds), 1)
            if avg_test < 5:
                lines.append("")
                lines.append(f"Confidence may be low: avg test rows per fold = {avg_test:.2f} (small holdout).")

        if self._last_benchmark_uncertainty_enabled:
            def _fmt(v: Optional[float]) -> str:
                if v is None:
                    return "-"
                return f"{float(v):.3f}"

            lines.append("")
            lines.append("Interval quality (conformal):")
            lines.append(
                f"- RS: coverage={_fmt(best_run.summary.rs_interval_coverage_mean)}; "
                f"mean width={_fmt(best_run.summary.rs_interval_mean_width_mean)}; "
                f"median width={_fmt(best_run.summary.rs_interval_median_width_mean)}"
            )
            lines.append(
                f"- Thickness: coverage={_fmt(best_run.summary.thickness_interval_coverage_mean)}; "
                f"mean width={_fmt(best_run.summary.thickness_interval_mean_width_mean)}; "
                f"median width={_fmt(best_run.summary.thickness_interval_median_width_mean)}"
            )
            lines.append(
                f"- RSU: coverage={_fmt(best_run.summary.rsu_interval_coverage_mean)}; "
                f"mean width={_fmt(best_run.summary.rsu_interval_mean_width_mean)}; "
                f"median width={_fmt(best_run.summary.rsu_interval_median_width_mean)}"
            )

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

    def _start_backtest_worker(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
        selected_bundles: list[str],
        split_mode: str,
    ) -> None:
        if self._backtest_thread is not None and self._backtest_thread.isRunning():
            self.bench_status_label.setText("Recommendation backtest already running...")
            return

        self.bench_run_button.setEnabled(False)
        self.bench_status_label.setText("Running recommendation backtest...")
        self.bench_backtest_enabled_checkbox.setEnabled(False)
        self.bench_backtest_abort_button.setEnabled(True)
        self.bench_backtest_progress.setValue(0)

        max_test_rows_per_fold = int(self.bench_backtest_max_test_rows_spin.value())
        self.bench_backtest_max_test_rows_spin.setEnabled(False)
        self._backtest_abort_event = threading.Event()

        self._backtest_thread = QThread(self)
        self._backtest_worker = _BacktestWorker(
            df=df.copy(),
            config=dict(config),
            model_bundle_names=list(selected_bundles),
            split_mode=split_mode,
            max_test_rows_per_fold=max_test_rows_per_fold,
            abort_event=self._backtest_abort_event,
        )
        self._backtest_worker.moveToThread(self._backtest_thread)
        self._backtest_thread.started.connect(self._backtest_worker.run)
        self._backtest_worker.finished.connect(self._on_backtest_finished)
        self._backtest_worker.failed.connect(self._on_backtest_failed)
        if hasattr(self._backtest_worker, "progress"):
            self._backtest_worker.progress.connect(self._on_backtest_progress)
        self._backtest_worker.finished.connect(self._backtest_thread.quit)
        self._backtest_worker.failed.connect(self._backtest_thread.quit)
        self._backtest_thread.finished.connect(self._cleanup_backtest_worker)
        self._backtest_thread.start()

    def _on_backtest_finished(self, result: Any) -> None:
        self._last_recommendation_backtest_result = result
        self._render_recommendation_backtest(result)
        self._refresh_bench_chart()
        self.bench_backtest_enabled_checkbox.setEnabled(True)
        self.bench_run_button.setEnabled(True)
        self.bench_backtest_abort_button.setEnabled(False)
        self.bench_backtest_max_test_rows_spin.setEnabled(self.bench_backtest_enabled_checkbox.isChecked())
        if getattr(result, "aborted", False):
            self.bench_status_label.setText("Recommendation backtest aborted.")
        else:
            self.bench_status_label.setText("Recommendation backtest completed.")

    def _on_backtest_failed(self, message: str) -> None:
        self._last_recommendation_backtest_result = None
        self.bench_backtest_enabled_checkbox.setEnabled(True)
        self.bench_run_button.setEnabled(True)
        self.bench_backtest_abort_button.setEnabled(False)
        self.bench_backtest_max_test_rows_spin.setEnabled(self.bench_backtest_enabled_checkbox.isChecked())
        self.bench_best_explanation.setPlainText(f"Recommendation backtest failed: {message}")
        self.bench_status_label.setText("Recommendation backtest failed.")

    def _cleanup_backtest_worker(self) -> None:
        if self._backtest_worker is not None:
            self._backtest_worker.deleteLater()
        self._backtest_worker = None
        if self._backtest_thread is not None:
            self._backtest_thread.deleteLater()
        self._backtest_thread = None
        self._backtest_abort_event = None
        self.bench_backtest_abort_button.setEnabled(False)
        self.bench_backtest_max_test_rows_spin.setEnabled(self.bench_backtest_enabled_checkbox.isChecked())


class _BacktestWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        config: dict[str, Any],
        model_bundle_names: list[str],
        split_mode: str,
        max_test_rows_per_fold: int,
        abort_event: Any,
    ) -> None:
        super().__init__()
        self._df = df
        self._config = config
        self._model_bundle_names = model_bundle_names
        self._split_mode = split_mode
        self._max_test_rows_per_fold = max_test_rows_per_fold
        self._abort_event = abort_event

    def run(self) -> None:
        try:
            from core.recommendation_backtest import run_recommendation_backtest

            def _progress_cb(done: int, total: int, stage: str) -> None:
                pct = int((done / total) * 100.0) if total else 0
                # stage can be long; keep it as-is for tooltip-ish status.
                self.progress.emit(pct, stage)

            result = run_recommendation_backtest(
                self._df,
                config=self._config,
                model_bundle_names=self._model_bundle_names,
                split_mode=self._split_mode,
                max_test_rows_per_fold=self._max_test_rows_per_fold,
                abort_event=self._abort_event,
                progress_cb=_progress_cb,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

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

