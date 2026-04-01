"""
Tabbed application settings dialog (parameters, aliases, minimum steps, placeholders).

Persists the structured parameter registry and user_config keys via callbacks.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_store import load_user_config
from core.parameter_registry import (
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
    load_effective_parameter_registry,
    save_user_parameter_registry,
)
from core.settings_editor import (
    apply_alias_pairs_to_registry,
    collect_alias_pairs,
    combo_text_from_data_type,
    data_type_from_combo_text,
    merge_registry_validation_messages,
    parse_optional_float,
    validate_alias_pairs,
    validate_duplicate_canonicals,
    validate_minimum_steps_dict,
)

_DATA_TYPE_LABELS = ("continuous", "discrete_step", "categorical", "boolean")
_DATA_TYPE_CHOICES = frozenset(_DATA_TYPE_LABELS)


class AppSettingsDialog(QDialog):
    """
    Multi-tab settings editor.

    ``on_apply`` receives updated fragments; the caller merges into user_config / reloads config.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        initial_user_config: dict[str, Any],
        on_apply: Callable[[dict[str, Any]], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._on_apply = on_apply
        self.setWindowTitle("Settings")
        self.setMinimumSize(920, 560)

        self._snapshot_user_config = deepcopy(initial_user_config)
        self._dep_by_canonical: dict[str, Any] = {}
        self._param_rows: list[ParameterDefinition] = []

        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        self._validation_label.setStyleSheet("color: #a33;")

        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_general_tab(), "General")
        self.tabs.addTab(self._build_parameters_tab(), "Parameters")
        self.tabs.addTab(self._build_aliases_tab(), "Parameter Aliases / Excel Matching")
        self.tabs.addTab(self._build_dependent_tab(), "Dependent Rules")
        self.tabs.addTab(self._build_model_defaults_tab(), "Model / Validation Defaults")
        layout.addWidget(self.tabs)
        layout.addWidget(self._validation_label)

        btn_row = QHBoxLayout()
        self._btn_save = QPushButton("Save")
        self._btn_cancel = QPushButton("Cancel")
        self._btn_reset = QPushButton("Reset")
        self._btn_apply = QPushButton("Apply")
        for b in (self._btn_save, self._btn_cancel, self._btn_reset, self._btn_apply):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self._btn_save.clicked.connect(self._on_save_clicked)
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._btn_apply.clicked.connect(self._on_apply_clicked)

        self._load_from_disk()

    # --- tab builders ---

    def _build_general_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        info = QLabel(
            "User settings file and parameter registry are stored under the project "
            "<b>config</b> folder. Changes apply after Save or Apply."
        )
        info.setWordWrap(True)
        v.addWidget(info)
        paths = QLabel(
            f"<code>config/user_config.json</code><br/>"
            f"<code>config/user_parameter_registry.json</code>"
        )
        paths.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(paths)

        grp = QGroupBox("Minimum parameter steps")
        gv = QVBoxLayout()
        self._min_steps_table = QTableWidget(0, 2)
        self._min_steps_table.setHorizontalHeaderLabels(["Parameter", "Minimum step"])
        self._min_steps_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        gv.addWidget(self._min_steps_table)
        row_btns = QHBoxLayout()
        self._min_add = QPushButton("Add row")
        self._min_del = QPushButton("Remove selected")
        row_btns.addWidget(self._min_add)
        row_btns.addWidget(self._min_del)
        row_btns.addStretch(1)
        gv.addLayout(row_btns)
        self._min_add.clicked.connect(self._min_steps_add_row)
        self._min_del.clicked.connect(self._min_steps_remove_selected)
        grp.setLayout(gv)
        v.addWidget(grp)
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _build_parameters_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        self._param_table = QTableWidget(0, 9)
        self._param_table.setHorizontalHeaderLabels(
            [
                "Enabled",
                "Display name",
                "Canonical name",
                "Unit",
                "Type",
                "Min",
                "Max",
                "Step",
                "Notes",
            ]
        )
        hdr = self._param_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self._param_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._param_table.setAlternatingRowColors(True)
        self._param_table.itemChanged.connect(self._on_param_item_changed)

        pv = QVBoxLayout()
        pv.addWidget(self._param_table)
        pbtn = QHBoxLayout()
        self._p_add = QPushButton("Add parameter")
        self._p_edit = QPushButton("Edit details…")
        self._p_del = QPushButton("Delete")
        self._p_up = QPushButton("Move up")
        self._p_down = QPushButton("Move down")
        for b in (self._p_add, self._p_edit, self._p_del, self._p_up, self._p_down):
            pbtn.addWidget(b)
        pbtn.addStretch(1)
        pv.addLayout(pbtn)
        self._p_add.clicked.connect(self._param_add)
        self._p_edit.clicked.connect(self._param_edit_details)
        self._p_del.clicked.connect(self._param_delete)
        self._p_up.clicked.connect(lambda: self._param_move(-1))
        self._p_down.clicked.connect(lambda: self._param_move(1))

        v.addLayout(pv)
        w.setLayout(v)
        return w

    def _build_aliases_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        hint = QLabel(
            "Map Excel column headers to canonical parameter names. "
            "Several headers may point to the same canonical name."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        self._alias_table = QTableWidget(0, 2)
        self._alias_table.setHorizontalHeaderLabels(["Excel header", "Canonical parameter"])
        self._alias_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._alias_table.setAlternatingRowColors(True)
        av = QHBoxLayout()
        self._a_add = QPushButton("Add mapping")
        self._a_del = QPushButton("Remove selected")
        av.addWidget(self._a_add)
        av.addWidget(self._a_del)
        av.addStretch(1)
        self._a_add.clicked.connect(self._alias_add_row)
        self._a_del.clicked.connect(self._alias_remove_selected)
        v.addWidget(self._alias_table)
        v.addLayout(av)
        w.setLayout(v)
        return w

    def _build_dependent_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        v.addWidget(
            QLabel(
                "Dependent range rules (bounds that depend on another parameter) are shown below "
                "for reference. Full editing will be added in a later milestone."
            )
        )
        self._dependent_view = QTextEdit()
        self._dependent_view.setReadOnly(True)
        self._dependent_view.setPlaceholderText("No dependent rules in the current registry.")
        v.addWidget(self._dependent_view)
        w.setLayout(v)
        return w

    def _build_model_defaults_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout()
        self._split_combo = QComboBox()
        self._split_combo.addItems(["", "auto", "manual", "time_series"])
        self._cutoff_combo = QComboBox()
        self._cutoff_combo.addItems(["", "median", "mean", "quantile"])
        self._metrics_edit = QLineEdit()
        self._metrics_edit.setPlaceholderText("e.g. rmse, mae, r2 (comma-separated)")
        form.addRow("Default split mode:", self._split_combo)
        form.addRow("Default cutoff mode:", self._cutoff_combo)
        form.addRow("Default metrics to show:", self._metrics_edit)
        note = QLabel("These values are stored in user_config for future validation / model UI.")
        note.setWordWrap(True)
        form.addRow(note)
        w.setLayout(form)
        return w

    # --- load / reset ---

    def _load_from_disk(self) -> None:
        reg = load_effective_parameter_registry(self._project_root)
        self._dep_by_canonical = {
            name: deepcopy(d.dependent_constraints) for name, d in reg._by_canonical.items()
        }
        self._param_rows = [deepcopy(d) for d in reg._by_canonical.values()]

        uc = load_user_config(self._project_root)
        self._snapshot_user_config = deepcopy(uc)
        self._fill_min_steps_table(dict(uc.get("minimum_steps") or {}))
        self._fill_model_defaults(dict(uc.get("model_validation_defaults") or {}))
        self._rebuild_param_table()
        self._rebuild_alias_table_from_registry(self._registry_from_param_rows())
        self._refresh_dependent_view()
        self._clear_validation()

    def _fill_min_steps_table(self, steps: dict[str, Any]) -> None:
        self._min_steps_table.setRowCount(0)
        for k, v in sorted(steps.items(), key=lambda kv: str(kv[0]).lower()):
            self._min_steps_add_row(str(k), v)

    def _min_steps_add_row(self, name: str = "", value: Any = 0.0) -> None:
        r = self._min_steps_table.rowCount()
        self._min_steps_table.insertRow(r)
        self._min_steps_table.setItem(r, 0, QTableWidgetItem(str(name)))
        try:
            fv = float(value)
        except (TypeError, ValueError):
            fv = 0.0
        self._min_steps_table.setItem(r, 1, QTableWidgetItem(str(fv)))

    def _min_steps_remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._min_steps_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._min_steps_table.removeRow(r)

    def _fill_model_defaults(self, d: dict[str, Any]) -> None:
        sm = str(d.get("default_split_mode") or "")
        ix = self._split_combo.findText(sm)
        self._split_combo.setCurrentIndex(ix if ix >= 0 else 0)
        cm = str(d.get("default_cutoff_mode") or "")
        ix2 = self._cutoff_combo.findText(cm)
        self._cutoff_combo.setCurrentIndex(ix2 if ix2 >= 0 else 0)
        metrics = d.get("default_metrics")
        if isinstance(metrics, list):
            self._metrics_edit.setText(", ".join(str(x) for x in metrics))
        else:
            self._metrics_edit.setText("")

    def _rebuild_param_table(self) -> None:
        self._param_table.blockSignals(True)
        self._param_table.setRowCount(0)
        for i, d in enumerate(self._param_rows):
            self._param_table.insertRow(i)
            self._write_param_row(i, d)
        self._param_table.blockSignals(False)

    def _write_param_row(self, row: int, d: ParameterDefinition) -> None:
        en = QTableWidgetItem("")
        en.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        )
        en.setCheckState(Qt.CheckState.Checked if d.enabled else Qt.CheckState.Unchecked)
        self._param_table.setItem(row, 0, en)
        self._param_table.setItem(row, 1, QTableWidgetItem(d.display_name))
        self._param_table.setItem(row, 2, QTableWidgetItem(d.canonical_name))
        unit = d.unit if d.unit is not None else ""
        self._param_table.setItem(row, 3, QTableWidgetItem(str(unit)))
        type_item = QTableWidgetItem(combo_text_from_data_type(d.data_type))
        self._param_table.setItem(row, 4, type_item)
        emin, emax = d.effective_min(), d.effective_max()
        self._param_table.setItem(row, 5, QTableWidgetItem("" if emin is None else str(emin)))
        self._param_table.setItem(row, 6, QTableWidgetItem("" if emax is None else str(emax)))
        st = d.effective_step()
        self._param_table.setItem(row, 7, QTableWidgetItem(str(st)))
        self._param_table.setItem(row, 8, QTableWidgetItem(d.notes or ""))

    def _on_param_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row < 0 or row >= len(self._param_rows):
            return
        self._sync_param_row_from_table(row)

    def _sync_param_row_from_table(self, row: int) -> None:
        """Update the backing model from the table (lenient parsing while editing)."""
        d = self._param_rows[row]
        for col in range(9):
            if self._param_table.item(row, col) is None:
                return
        d.enabled = self._param_table.item(row, 0).checkState() == Qt.CheckState.Checked
        d.display_name = self._param_table.item(row, 1).text().strip()
        d.canonical_name = self._param_table.item(row, 2).text().strip()
        unit_txt = self._param_table.item(row, 3).text().strip()
        d.unit = unit_txt if unit_txt else None
        type_txt = self._param_table.item(row, 4).text().strip().lower()
        if type_txt in _DATA_TYPE_CHOICES:
            d.data_type = data_type_from_combo_text(type_txt)
        elif not type_txt:
            d.data_type = data_type_from_combo_text("continuous")
        try:
            d.min_value = parse_optional_float(self._param_table.item(row, 5).text())
            d.max_value = parse_optional_float(self._param_table.item(row, 6).text())
        except ValueError:
            pass
        try:
            st_txt = self._param_table.item(row, 7).text().strip()
            d.step = float(st_txt) if st_txt else 0.0
        except ValueError:
            pass
        notes_item = self._param_table.item(row, 8)
        d.notes = notes_item.text() if notes_item else ""
        d.simple_range = None
        self._refresh_dependent_view()

    def _collect_param_table_errors(self) -> list[str]:
        """Strict validation messages for parameter rows (reads the table directly)."""
        errors: list[str] = []
        if self._param_table.rowCount() != len(self._param_rows):
            errors.append("Parameter table is out of sync; try Reset.")
            return errors
        for row in range(self._param_table.rowCount()):
            pr = row + 1
            type_txt = (self._param_table.item(row, 4).text() if self._param_table.item(row, 4) else "").strip().lower()
            if not type_txt:
                errors.append(f"Row {pr}: type is required ({', '.join(sorted(_DATA_TYPE_CHOICES))}).")
            elif type_txt not in _DATA_TYPE_CHOICES:
                errors.append(f"Row {pr}: invalid type {type_txt!r}.")
            for col, label in ((5, "Min"), (6, "Max")):
                cell = self._param_table.item(row, col)
                raw = cell.text() if cell else ""
                try:
                    parse_optional_float(raw)
                except ValueError as e:
                    errors.append(f"Row {pr} {label}: {e}")
            st_cell = self._param_table.item(row, 7)
            st_raw = (st_cell.text() if st_cell else "").strip()
            if st_raw:
                try:
                    float(st_raw)
                except ValueError:
                    errors.append(f"Row {pr}: step is not a valid number.")
        return errors

    def _registry_from_param_rows(self) -> ParameterRegistry:
        reg = ParameterRegistry()
        for d in self._param_rows:
            dd = deepcopy(d)
            if dd.canonical_name in self._dep_by_canonical:
                dd.dependent_constraints = deepcopy(self._dep_by_canonical[dd.canonical_name])
            else:
                dd.dependent_constraints = []
            reg.add(dd)
        return reg

    def _rebuild_alias_table_from_registry(self, reg: ParameterRegistry) -> None:
        pairs = collect_alias_pairs(reg)
        self._alias_table.setRowCount(0)
        for h, c in pairs:
            r = self._alias_table.rowCount()
            self._alias_table.insertRow(r)
            self._alias_table.setItem(r, 0, QTableWidgetItem(h))
            self._alias_table.setItem(r, 1, QTableWidgetItem(c))

    def _get_alias_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for r in range(self._alias_table.rowCount()):
            h_item = self._alias_table.item(r, 0)
            c_item = self._alias_table.item(r, 1)
            h = h_item.text() if h_item else ""
            c = c_item.text() if c_item else ""
            pairs.append((h, c))
        return pairs

    def _alias_add_row(self) -> None:
        r = self._alias_table.rowCount()
        self._alias_table.insertRow(r)
        self._alias_table.setItem(r, 0, QTableWidgetItem(""))
        self._alias_table.setItem(r, 1, QTableWidgetItem(""))

    def _alias_remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._alias_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._alias_table.removeRow(r)

    def _refresh_dependent_view(self) -> None:
        reg = self._registry_from_param_rows()
        apply_alias_pairs_to_registry(reg, self._get_alias_pairs())
        lines: list[str] = []
        for name in sorted(reg._by_canonical.keys()):
            d = reg.get_by_canonical(name)
            if d and d.dependent_constraints:
                lines.append(f"{name}:")
                for dc in d.dependent_constraints:
                    lines.append(f"  driver: {dc.driver_parameter}")
                    for rule in dc.rules:
                        lines.append(
                            f"    when driver in [{rule.driver_min}, {rule.driver_max}] "
                            f"-> bounds [{rule.min_value}, {rule.max_value}]"
                        )
        self._dependent_view.setPlainText("\n".join(lines) if lines else "")

    # --- parameter row ops ---

    def _param_add(self) -> None:
        new_name = f"new_param_{len(self._param_rows) + 1}"
        d = ParameterDefinition(
            canonical_name=new_name,
            display_name="New parameter",
            data_type=data_type_from_combo_text("continuous"),
            min_value=None,
            max_value=None,
            step=0.0,
            enabled=True,
            notes="",
        )
        self._param_rows.append(d)
        r = len(self._param_rows) - 1
        self._param_table.insertRow(r)
        self._write_param_row(r, d)
        self._refresh_dependent_view()

    def _param_edit_details(self) -> None:
        row = self._param_table.currentRow()
        if row < 0 or row >= len(self._param_rows):
            QMessageBox.information(self, "Edit details", "Select a parameter row first.")
            return
        d = self._param_rows[row]
        dlg = QDialog(self)
        dlg.setWindowTitle("Parameter details")
        form = QFormLayout()
        allowed_edit = QLineEdit()
        if d.allowed_values is not None:
            allowed_edit.setText(", ".join(str(x) for x in d.allowed_values))
        coupled = QCheckBox("Coupled parameter")
        coupled.setChecked(d.is_coupled)
        group_edit = QLineEdit(d.coupling_group or "")
        form.addRow("Allowed values (comma-separated, for categorical):", allowed_edit)
        form.addRow(coupled)
        form.addRow("Coupling group:", group_edit)
        bb = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        bb.addWidget(ok)
        bb.addWidget(cancel)
        form.addRow(bb)
        dlg.setLayout(form)

        def accept() -> None:
            raw = allowed_edit.text().strip()
            parts = [p.strip() for p in raw.split(",") if p.strip()] if raw else []
            if d.data_type == ParameterDataType.CATEGORICAL:
                d.allowed_values = parts if parts else None
            else:
                d.allowed_values = None
            d.is_coupled = coupled.isChecked()
            gg = group_edit.text().strip()
            d.coupling_group = gg if gg else None
            dlg.accept()

        ok.clicked.connect(accept)
        cancel.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._write_param_row(row, d)

    def _param_delete(self) -> None:
        rows = sorted({i.row() for i in self._param_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self._param_rows):
                del self._param_rows[r]
        self._rebuild_param_table()
        self._refresh_dependent_view()

    def _param_move(self, delta: int) -> None:
        row = self._param_table.currentRow()
        if row < 0:
            return
        n = row + delta
        if n < 0 or n >= len(self._param_rows):
            return
        self._param_rows[row], self._param_rows[n] = self._param_rows[n], self._param_rows[row]
        self._rebuild_param_table()
        self._param_table.selectRow(n)

    # --- validation / persist ---

    def _gather_minimum_steps(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in range(self._min_steps_table.rowCount()):
            k_item = self._min_steps_table.item(r, 0)
            v_item = self._min_steps_table.item(r, 1)
            k = (k_item.text() if k_item else "").strip()
            if not k:
                continue
            try:
                out[k] = float((v_item.text() if v_item else "0").strip() or 0)
            except ValueError:
                raise ValueError(f"Minimum steps row {r + 1}: step is not a valid number.") from None
        return out

    def _gather_model_defaults(self) -> dict[str, Any]:
        sm = self._split_combo.currentText().strip()
        cm = self._cutoff_combo.currentText().strip()
        raw_m = self._metrics_edit.text().strip()
        metrics = [p.strip() for p in raw_m.split(",") if p.strip()] if raw_m else []
        return {
            "default_split_mode": sm,
            "default_cutoff_mode": cm,
            "default_metrics": metrics,
        }

    def _sync_all_param_rows_from_table(self) -> None:
        for r in range(len(self._param_rows)):
            self._sync_param_row_from_table(r)

    def _validate(self) -> list[str]:
        errors: list[str] = []
        self._clear_validation()
        errors.extend(self._collect_param_table_errors())
        if errors:
            return errors
        self._sync_all_param_rows_from_table()
        names = [d.canonical_name for d in self._param_rows]
        errors.extend(validate_duplicate_canonicals(names))
        reg = self._registry_from_param_rows()
        apply_alias_pairs_to_registry(reg, self._get_alias_pairs())
        errors.extend(merge_registry_validation_messages(reg))
        canonicals = set(reg._by_canonical.keys())
        errors.extend(validate_alias_pairs(self._get_alias_pairs(), canonicals))
        try:
            ms = self._gather_minimum_steps()
            errors.extend(validate_minimum_steps_dict(ms))
        except ValueError as e:
            errors.append(str(e))
        return errors

    def _set_validation(self, message: str) -> None:
        self._validation_label.setText(message)

    def _clear_validation(self) -> None:
        self._validation_label.setText("")

    def _persist_payload(self) -> dict[str, Any]:
        """Build user_config fragment + save registry file."""
        self._sync_all_param_rows_from_table()
        reg = self._registry_from_param_rows()
        apply_alias_pairs_to_registry(reg, self._get_alias_pairs())
        save_user_parameter_registry(self._project_root, reg)
        ms = self._gather_minimum_steps()
        mvd = self._gather_model_defaults()
        return {
            "minimum_steps": ms,
            "model_validation_defaults": mvd,
        }

    def _on_apply_clicked(self) -> None:
        errs = self._validate()
        if errs:
            self._set_validation("\n".join(errs))
            return
        self._clear_validation()
        payload = self._persist_payload()
        self._on_apply(payload)
        self._snapshot_user_config = deepcopy({**self._snapshot_user_config, **payload})

    def _on_save_clicked(self) -> None:
        errs = self._validate()
        if errs:
            self._set_validation("\n".join(errs))
            return
        self._clear_validation()
        payload = self._persist_payload()
        self._on_apply(payload)
        self.accept()

    def _on_reset_clicked(self) -> None:
        self._load_from_disk()
