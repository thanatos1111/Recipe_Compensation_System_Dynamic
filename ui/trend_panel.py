"""
Trend analysis panel (Milestone 3).

Plots key columns vs lifetime, distinguishes in-spec vs out-of-spec, and
shows a baseline reference table by lifetime bins.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import QComboBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.schemas import MaterialDataset
from core.trend_fitting import build_lifetime_reference_table

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    FigureCanvas = None  # type: ignore[assignment]

from matplotlib.figure import Figure


class TrendPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._material_dataset: Optional[MaterialDataset] = None
        self._active_target_id: Optional[str] = None
        self._config: dict[str, Any] = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Trend panel (Milestone 3)"))

        layout.addWidget(QLabel("Y-axis selection:"))
        self.y_selector = QComboBox()
        self.y_selector.addItems(
            [
                "incident_angle",
                "linear_offset",
                "rotations",
                "ar_flow",
                "o2_flow",
                "RS",
                "Thickness",
                "RSU",
            ]
        )
        self.y_selector.currentTextChanged.connect(self._refresh_plot)
        layout.addWidget(self.y_selector)

        self.figure = Figure(figsize=(6, 3))
        self.ax = self.figure.add_subplot(111)

        if FigureCanvas is not None:
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            layout.setStretchFactor(self.canvas, 3)
        else:
            layout.addWidget(QLabel("Matplotlib Qt canvas not available."))
            self.canvas = None

        layout.addWidget(QLabel("Baseline reference (lifetime bins):"))
        self.baseline_table = QTableWidget()
        layout.addWidget(self.baseline_table)
        layout.setStretchFactor(self.baseline_table, 1)

        self.setLayout(layout)

    def set_context(self, material_dataset: MaterialDataset, *, active_target_id: str, config: dict[str, Any]) -> None:
        self._material_dataset = material_dataset
        self._active_target_id = active_target_id
        self._config = config
        self._refresh_plot()

    def _y_column(self, y_sel: str) -> str:
        if y_sel == "RS":
            return "rs"
        if y_sel == "Thickness":
            return "thickness"
        if y_sel == "RSU":
            return "rsu"
        return y_sel

    def _refresh_plot(self) -> None:
        if self._material_dataset is None or not self._active_target_id:
            return

        y_sel = self.y_selector.currentText()
        y_col = self._y_column(y_sel)
        x_col = "lifetime"

        self.ax.clear()

        for target_id, inst in self._material_dataset.target_instances.items():
            records = inst.records
            if records is None or records.empty:
                continue
            if x_col not in records.columns or y_col not in records.columns:
                continue

            if "in_spec" in records.columns:
                in_spec_mask = records["in_spec"] == True  # noqa: E712
            else:
                in_spec_mask = records.index.to_series().apply(lambda _: True)

            x_in = records.loc[in_spec_mask, x_col]
            y_in = records.loc[in_spec_mask, y_col]
            x_out = records.loc[~in_spec_mask, x_col]
            y_out = records.loc[~in_spec_mask, y_col]

            self.ax.scatter(x_in, y_in, s=22, marker="o", alpha=0.9, label=f"{target_id} in-spec")
            if len(x_out) > 0:
                # Out-of-spec points: more visible marker so they aren't mistaken for "missing".
                self.ax.scatter(x_out, y_out, s=26, marker="x", alpha=0.8)

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.legend(fontsize=8, loc="best")
        self.ax.grid(True, alpha=0.2)
        self.figure.tight_layout()

        if self.canvas is not None:
            self.canvas.draw_idle()

        active_inst = self._material_dataset.target_instances.get(self._active_target_id)
        if active_inst is None:
            return
        baseline_df = build_lifetime_reference_table(active_inst.records, self._config)
        self._render_baseline_table(baseline_df)

    def _render_baseline_table(self, baseline_df: Any) -> None:
        if baseline_df is None or baseline_df.empty:
            self.baseline_table.setRowCount(0)
            self.baseline_table.setColumnCount(0)
            return

        self.baseline_table.clear()
        self.baseline_table.setRowCount(len(baseline_df))
        self.baseline_table.setColumnCount(len(baseline_df.columns))

        for c, col_name in enumerate(baseline_df.columns):
            self.baseline_table.setHorizontalHeaderItem(c, QTableWidgetItem(str(col_name)))

        for r in range(len(baseline_df)):
            for c, col_name in enumerate(baseline_df.columns):
                value = baseline_df.iloc[r][col_name]
                # Handle NaN to keep cells empty.
                try:
                    is_nan = value != value
                except Exception:
                    is_nan = False
                text = "" if value is None or is_nan else str(value)
                self.baseline_table.setItem(r, c, QTableWidgetItem(text))

