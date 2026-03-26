"""
Raw + labeled rows table panel (Milestone 3).

Displays canonical normalized data after spec labeling (adds `in_spec` and
derived columns), with UI filtering for target instance and in-spec status.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RawTablePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.info_label = QLabel("Raw table (Milestone 3): no data loaded.")

        self._df_full: Optional[pd.DataFrame] = None
        self._target_ids: list[str] = []
        self._active_target_id: Optional[str] = None

        self.target_filter_combo = QComboBox()
        self.target_filter_combo.addItem("All targets")
        self.target_filter_combo.setEnabled(False)

        self.in_spec_filter_combo = QComboBox()
        self.in_spec_filter_combo.addItems(["All", "In-spec only", "Out-of-spec only"])
        self.in_spec_filter_combo.setEnabled(False)

        self.target_filter_combo.currentTextChanged.connect(self._refresh_table)
        self.in_spec_filter_combo.currentTextChanged.connect(self._refresh_table)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Raw + labeled rows (Milestone 3)"))
        layout.addWidget(QLabel("Target filter:"))
        layout.addWidget(self.target_filter_combo)
        layout.addWidget(QLabel("In-spec filter:"))
        layout.addWidget(self.in_spec_filter_combo)
        layout.addWidget(self.info_label)
        layout.addWidget(self.table)

        # Give the table most of the vertical space within the tab.
        layout.setStretchFactor(self.table, 1)

        self.setLayout(layout)

    def set_target_ids(self, target_ids: list[str], *, active_target_id: Optional[str] = None) -> None:
        self._target_ids = list(target_ids)
        self._active_target_id = active_target_id

        self.target_filter_combo.blockSignals(True)
        self.target_filter_combo.clear()
        self.target_filter_combo.addItem("All targets")
        for tid in self._target_ids:
            self.target_filter_combo.addItem(tid)
        self.target_filter_combo.setEnabled(bool(self._target_ids))

        if active_target_id and active_target_id in self._target_ids:
            self.target_filter_combo.setCurrentText(active_target_id)
        else:
            self.target_filter_combo.setCurrentIndex(0)

        self.target_filter_combo.blockSignals(False)

        self.in_spec_filter_combo.setEnabled(True)
        self._refresh_table()

    def set_material_dataframe(self, df: pd.DataFrame) -> None:
        """Set the labeled dataframe for the selected material."""
        self._df_full = df
        self._refresh_table()

    def set_active_target_id(self, active_target_id: str) -> None:
        self._active_target_id = active_target_id
        if self.target_filter_combo.isEnabled():
            self.target_filter_combo.blockSignals(True)
            self.target_filter_combo.setCurrentText(active_target_id)
            self.target_filter_combo.blockSignals(False)
        self._refresh_table()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Back-compat: treat `df` as a single-target view.

        Milestone 3 prefers `set_material_dataframe`, but older wiring may still
        call `set_dataframe`.
        """
        self._df_full = df
        self._target_ids = []
        self._active_target_id = None
        self.target_filter_combo.setEnabled(False)
        self.in_spec_filter_combo.setEnabled(True)
        self.target_filter_combo.setCurrentIndex(0)
        self._refresh_table()

    def _refresh_table(self) -> None:
        df = self._df_full
        if df is None or df.empty:
            self.info_label.setText("Raw table: no rows loaded.")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        # Filter by target.
        selected_target = (
            self.target_filter_combo.currentText() if self.target_filter_combo.isEnabled() else "All targets"
        )
        if selected_target and selected_target != "All targets" and "target_id" in df.columns:
            df = df[df["target_id"] == selected_target]

        # Filter by in-spec.
        in_spec_mode = self.in_spec_filter_combo.currentText()
        if "in_spec" in df.columns:
            if in_spec_mode == "In-spec only":
                df = df[df["in_spec"] == True]  # noqa: E712
            elif in_spec_mode == "Out-of-spec only":
                df = df[df["in_spec"] == False]  # noqa: E712

        self.info_label.setText(f"Showing {len(df)} rows; {len(df.columns)} columns.")

        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))

        # Headers
        for c, col_name in enumerate(df.columns):
            item = QTableWidgetItem(str(col_name))
            self.table.setHorizontalHeaderItem(c, item)

        # Cells
        for r in range(len(df)):
            for c, col_name in enumerate(df.columns):
                value = df.iloc[r][col_name]
                text = "" if pd.isna(value) else str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))

