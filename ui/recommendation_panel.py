"""
Recommendation panel (Milestone 5).

Generates discrete recipe candidates around the latest actual recipe,
predicts outputs using trained models + instance correction, ranks candidates,
and shows the best next recipe plus safe-band intervals.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSplitter,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    FigureCanvas = None  # type: ignore[assignment]

from matplotlib.figure import Figure


class RecommendationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._material_dataset = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}
        self._last_ranked: Optional[pd.DataFrame] = None

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

        self.generate_button = QPushButton("Generate recommendation")
        self.generate_button.clicked.connect(self._generate_clicked)
        layout.addWidget(self.generate_button)

        self.content_tabs = QTabWidget()
        layout.addWidget(self.content_tabs)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        overview_layout.addWidget(QLabel("Recommended next recipe:"))
        self.recommended_label = QLabel("")
        self.recommended_label.setWordWrap(True)
        overview_layout.addWidget(self.recommended_label)
        self.recommendation_prediction_label = QLabel("")
        self.recommendation_prediction_label.setWordWrap(True)
        overview_layout.addWidget(self.recommendation_prediction_label)
        overview_layout.addWidget(QLabel("Ranked candidates (top):"))
        self.candidates_table = QTableWidget()
        self.candidates_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        overview_layout.addWidget(self.candidates_table)
        overview_layout.setStretchFactor(self.candidates_table, 1)
        overview_tab.setLayout(overview_layout)
        self.content_tabs.addTab(overview_tab, "Overview")

        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout()
        # Left/right layout so the comparison table + plots get enough vertical
        # space to be readable.
        comparison_splitter = QSplitter(Qt.Orientation.Horizontal)
        comparison_splitter.setChildrenCollapsible(False)

        # Top pane: candidate selector + score breakdown.
        top_panel = QWidget()
        top_layout = QVBoxLayout()
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Selected candidate (compare with best):"))
        self.candidate_selector = QComboBox()
        self.candidate_selector.currentIndexChanged.connect(self._on_candidate_selection_changed)
        compare_row.addWidget(self.candidate_selector)
        compare_row.addStretch(1)
        top_layout.addLayout(compare_row)
        top_layout.addWidget(QLabel("Score breakdown (best candidate):"))
        self.score_breakdown_table = QTableWidget()
        self.score_breakdown_table.setWordWrap(True)
        self.score_breakdown_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.score_breakdown_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.score_breakdown_table.setMinimumHeight(170)
        top_layout.addWidget(self.score_breakdown_table)
        top_layout.setStretchFactor(self.score_breakdown_table, 1)
        top_panel.setLayout(top_layout)

        comparison_views_tabs = QTabWidget()

        table_view = QWidget()
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("Top-N candidate comparison table"))
        self.comparison_table = QTableWidget()
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        self.comparison_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.comparison_table.setMinimumHeight(260)
        table_layout.addWidget(self.comparison_table)
        table_layout.setStretchFactor(self.comparison_table, 1)
        table_view.setLayout(table_layout)
        comparison_views_tabs.addTab(table_view, "Comparison Table")

        plot_view = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(QLabel("Candidate score and parameter-output delta plots"))
        self.figure = Figure(figsize=(8, 4.6))
        self.ax_candidates = self.figure.add_subplot(121)
        self.ax_deltas = self.figure.add_subplot(122)
        if FigureCanvas is not None:
            self.canvas = FigureCanvas(self.figure)
            # Keep canvas responsive so the bottom pane doesn't get pushed off-screen.
            self.canvas.setMinimumHeight(230)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            plot_layout.addWidget(self.canvas)
            plot_layout.setStretchFactor(self.canvas, 1)
        else:
            self.canvas = None
            plot_layout.addWidget(QLabel("Matplotlib Qt canvas not available."))
        plot_view.setLayout(plot_layout)
        comparison_views_tabs.addTab(plot_view, "Plots")

        # Bottom pane: comparison table + plots subtabs.
        comparison_splitter.addWidget(top_panel)

        # The bottom pane can get tall (table + plots). Wrap it in a scroll
        # container so content doesn't overflow the tab on small windows.
        comparison_views_scroll = QScrollArea()
        comparison_views_scroll.setWidgetResizable(True)
        comparison_views_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        comparison_views_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        comparison_views_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        comparison_views_scroll.setWidget(comparison_views_tabs)
        comparison_splitter.addWidget(comparison_views_scroll)

        # Give the right pane (comparison tables/plots) most width.
        comparison_splitter.setStretchFactor(0, 1)
        comparison_splitter.setStretchFactor(1, 4)
        comparison_layout.addWidget(comparison_splitter)
        comparison_tab.setLayout(comparison_layout)
        self.content_tabs.addTab(comparison_tab, "Candidate Comparison")

        safe_band_tab = QWidget()
        safe_band_layout = QVBoxLayout()
        safe_band_layout.addWidget(QLabel("Safe-band explanation:"))
        self.safe_band_explanation_label = QLabel("")
        self.safe_band_explanation_label.setWordWrap(True)
        safe_band_layout.addWidget(self.safe_band_explanation_label)
        safe_band_layout.addWidget(QLabel("Safe-band details (per parameter):"))
        self.safe_band_label = QLabel("")
        self.safe_band_label.setWordWrap(True)
        self.safe_band_label.setTextInteractionFlags(self.safe_band_label.textInteractionFlags())
        safe_band_layout.addWidget(self.safe_band_label)
        safe_band_layout.addStretch(1)
        safe_band_tab.setLayout(safe_band_layout)
        self.content_tabs.addTab(safe_band_tab, "Safe Band")

        layout.setStretchFactor(self.content_tabs, 1)

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
        from core.safe_band import estimate_multi_parameter_bands

        parameter_config = self._config.get("parameter_constraints", {})

        def default_min_step(param_name: str) -> float:
            return 1.0 if param_name == "rotations" else 0.01

        # Ensure candidate generation + quantization always have a usable step.
        effective_parameter_config: dict[str, Any] = {}
        for pname, pcfg in (parameter_config or {}).items():
            if not isinstance(pcfg, dict):
                effective_parameter_config[pname] = pcfg
                continue
            eff = dict(pcfg)
            current_step = float(eff.get("step", 0.0) or 0.0)
            if current_step <= 0:
                eff["step"] = default_min_step(pname)
            else:
                eff["step"] = current_step
            effective_parameter_config[pname] = eff
        neighborhood_config = {
            "radius_steps": int(self.radius_steps_spin.value()),
            "max_candidates": int(self.max_candidates_spin.value()),
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

        self._last_ranked = ranked.copy()
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
        self.recommendation_prediction_label.setText(
            "Predicted outputs: RS={rs}, Thickness={th}, RSU={rsu} | score={score}".format(
                rs=best.get("rs_pred", None),
                th=best.get("thickness_pred", None),
                rsu=best.get("rsu_pred", None),
                score=best.get("score", None),
            )
        )

        self._refresh_recommendation_visualization(selected_index=0)

        safe_band_settings = self._config.get("safe_band_settings", {}) or {}
        safe_band_parameters = safe_band_settings.get("parameters")
        if not isinstance(safe_band_parameters, list) or not safe_band_parameters:
            safe_band_parameters = [
                p
                for p, cfg in (effective_parameter_config or {}).items()
                if isinstance(cfg, dict) and cfg.get("is_enabled", True)
            ]
        safe_band_parameters = [str(p) for p in safe_band_parameters if p in best]
        if not safe_band_parameters:
            safe_band_parameters = ["incident_angle"]

        safe_band_search_width_steps = int(safe_band_settings.get("search_width_steps", 3) or 3)
        clip_to_candidate_neighborhood = bool(safe_band_settings.get("clip_to_candidate_neighborhood", False))

        parameter_steps: dict[str, float] = {}
        for p in safe_band_parameters:
            if p in effective_parameter_config and isinstance(effective_parameter_config[p], dict):
                parameter_steps[p] = float(effective_parameter_config[p].get("step", 0.0) or 0.0)
            else:
                parameter_steps[p] = 1.0 if p == "rotations" else 0.01

        # Center around recommended candidate (best), with process context copied in.
        center_recipe = {k: best.get(k, reference_recipe.get(k, None)) for k in reference_recipe.keys()}
        center_recipe.update(
            {
                "target_id": self._active_target_id,
                "lifetime": lifetime,
                "rpm": rpm,
                "power": power,
            }
        )
        multi_bands = estimate_multi_parameter_bands(
            center_recipe,
            model_bundle,
            parameter_names=safe_band_parameters,
            spec_config=spec_config,
            parameter_steps=parameter_steps,
            candidate_radius_steps=int(neighborhood_config["radius_steps"]),
            safe_band_max_steps=safe_band_search_width_steps,
            clip_to_candidate_range=clip_to_candidate_neighborhood,
        )

        explanation_lines = [
            "Safe-band method: one-parameter-at-a-time predicted feasibility scan around recommended center values; all other parameters are fixed.",
            f"Candidate search neighborhood: +/- {int(neighborhood_config['radius_steps'])} step(s) per parameter.",
            f"Safe-band search width: +/- {safe_band_search_width_steps} step(s) per parameter.",
            f"Clip feasible band to candidate neighborhood: {'yes' if clip_to_candidate_neighborhood else 'no'}.",
        ]
        self.safe_band_explanation_label.setText("\n".join(explanation_lines))

        detail_lines: list[str] = []
        for p in safe_band_parameters:
            item = multi_bands.get(p, {})
            center_v = item.get("center_value")
            cand_range = item.get("candidate_search_range")
            feas_band = item.get("feasible_band")
            if cand_range is None or center_v is None:
                continue
            if feas_band is None:
                detail_lines.append(
                    f"{p}: recommended center value={center_v:.6g}; candidate search range=[{cand_range[0]:.6g}, {cand_range[1]:.6g}]; feasible band=none"
                )
            else:
                detail_lines.append(
                    f"{p}: recommended center value={center_v:.6g}; candidate search range=[{cand_range[0]:.6g}, {cand_range[1]:.6g}]; feasible band=[{feas_band[0]:.6g}, {feas_band[1]:.6g}]"
                )
        self.safe_band_label.setText("\n".join(detail_lines) if detail_lines else "No safe-band parameters configured.")

    def _on_candidate_selection_changed(self, index: int) -> None:
        self._refresh_recommendation_visualization(selected_index=index)

    def _refresh_recommendation_visualization(self, *, selected_index: int) -> None:
        if self._last_ranked is None or self._last_ranked.empty:
            return

        from core.recommendation_visualization import build_recommendation_visualization_payload

        payload = build_recommendation_visualization_payload(
            self._last_ranked,
            top_n=min(10, len(self._last_ranked)),
            selected_index=selected_index,
            parameter_order=["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"],
        )
        self._populate_candidate_selector(payload)
        self._render_score_breakdown(payload)
        self._render_top_n_comparison(payload)
        self._render_plots(payload)

    def _populate_candidate_selector(self, payload: dict[str, Any]) -> None:
        top_n_count = int(payload.get("top_n_count", 0) or 0)
        selected_index = int(payload.get("selected_index", 0) or 0)
        self.candidate_selector.blockSignals(True)
        self.candidate_selector.clear()
        for i in range(top_n_count):
            if i == 0:
                self.candidate_selector.addItem("Rank 1 (Best)")
            else:
                self.candidate_selector.addItem(f"Rank {i + 1}")
        if top_n_count > 0:
            self.candidate_selector.setCurrentIndex(max(0, min(selected_index, top_n_count - 1)))
        self.candidate_selector.blockSignals(False)

    def _render_score_breakdown(self, payload: dict[str, Any]) -> None:
        scores = payload.get("score_breakdown", {}) or {}
        keys = [k for k in ["score", "rs_error_norm", "thickness_error_norm", "rsu_penalty", "move_penalty", "safe_margin_reward"] if k in scores]
        self.score_breakdown_table.clear()
        self.score_breakdown_table.setRowCount(len(keys))
        self.score_breakdown_table.setColumnCount(2)
        self.score_breakdown_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.score_breakdown_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.score_breakdown_table.setHorizontalHeaderItem(0, QTableWidgetItem("Component"))
        self.score_breakdown_table.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        for ri, key in enumerate(keys):
            self.score_breakdown_table.setItem(ri, 0, QTableWidgetItem(key))
            val = scores.get(key)
            text = "-" if val is None else f"{float(val):.6g}"
            self.score_breakdown_table.setItem(ri, 1, QTableWidgetItem(text))
        self.score_breakdown_table.resizeRowsToContents()

    def _render_top_n_comparison(self, payload: dict[str, Any]) -> None:
        rows = payload.get("top_n_table", []) or []
        if not rows:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            return
        columns = list(rows[0].keys())
        self.comparison_table.clear()
        self.comparison_table.setRowCount(len(rows))
        self.comparison_table.setColumnCount(len(columns) + 1)
        self.comparison_table.setHorizontalHeaderItem(0, QTableWidgetItem("rank"))
        for ci, col in enumerate(columns, start=1):
            self.comparison_table.setHorizontalHeaderItem(ci, QTableWidgetItem(str(col)))
        selected_index = int(payload.get("selected_index", 0) or 0)
        for ri, row in enumerate(rows):
            rank_text = "1 (best)"
            if ri > 0:
                rank_text = f"{ri + 1}"
            if ri == selected_index and ri != 0:
                rank_text += " (selected)"
            self.comparison_table.setItem(ri, 0, QTableWidgetItem(rank_text))
            for ci, col in enumerate(columns, start=1):
                v = row.get(col)
                text = "" if v is None else str(v)
                self.comparison_table.setItem(ri, ci, QTableWidgetItem(text))
        self.comparison_table.resizeRowsToContents()

    def _render_plots(self, payload: dict[str, Any]) -> None:
        if self.canvas is None:
            return

        top_rows = payload.get("top_n_table", []) or []
        selected_index = int(payload.get("selected_index", 0) or 0)
        deltas = payload.get("parameter_prediction_deltas", []) or []

        self.ax_candidates.clear()
        self.ax_deltas.clear()

        if top_rows:
            rank = list(range(1, len(top_rows) + 1))
            scores = [float(r.get("score", 0.0) or 0.0) for r in top_rows]
            bars = self.ax_candidates.bar(rank, scores, color=["#4c78a8"] * len(rank))
            bars[0].set_color("#59a14f")
            if 0 <= selected_index < len(bars):
                bars[selected_index].set_color("#f28e2b")
            self.ax_candidates.set_title("Top-N score comparison")
            self.ax_candidates.set_xlabel("Candidate rank")
            self.ax_candidates.set_ylabel("Score (lower is better)")
            self.ax_candidates.grid(True, alpha=0.2, axis="y")

        if deltas:
            labels = [str(d.get("parameter", "")) for d in deltas]
            x = list(range(len(labels)))
            width = 0.25
            rs_vals = [float(d.get("delta_rs_pred", 0.0) or 0.0) for d in deltas]
            th_vals = [float(d.get("delta_thickness_pred", 0.0) or 0.0) for d in deltas]
            rsu_vals = [float(d.get("delta_rsu_pred", 0.0) or 0.0) for d in deltas]
            self.ax_deltas.bar([v - width for v in x], rs_vals, width=width, label="Delta RS")
            self.ax_deltas.bar(x, th_vals, width=width, label="Delta Thickness")
            self.ax_deltas.bar([v + width for v in x], rsu_vals, width=width, label="Delta RSU")
            self.ax_deltas.axhline(0.0, color="#666666", linewidth=1)
            self.ax_deltas.set_xticks(x)
            self.ax_deltas.set_xticklabels(labels, rotation=20)
            self.ax_deltas.set_title("Parameter change vs predicted output change")
            self.ax_deltas.set_ylabel("Selected - Best")
            self.ax_deltas.grid(True, alpha=0.2, axis="y")
            self.ax_deltas.legend(fontsize=7, loc="best")

        self.figure.tight_layout()
        self.canvas.draw_idle()

