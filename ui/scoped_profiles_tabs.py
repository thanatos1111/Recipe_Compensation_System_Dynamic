"""
Editors for scoped parameter and D.Spec profiles (Settings dialog).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.scoped_settings import (
    TARGET_POSITION_MAX,
    TARGET_POSITION_MIN,
    is_shipped_target_position_preset,
    validate_scoped_parameter_profile,
    validate_scoped_spec_profile,
)


def _safe_float(s: str) -> Optional[float]:
    t = (s or "").strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


class ScopedParameterProfilesEditor(QWidget):
    """List + form for ``parameter_profiles`` entries."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._profiles: list[dict[str, Any]] = []
        self._building = False
        self._prev_row = -1

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._id_edit = QLineEdit()
        self._enabled = QCheckBox("Enabled")
        self._enabled.setChecked(True)
        self._material = QLineEdit()
        self._material.setPlaceholderText("Blank = any material (use with target position)")
        self._tp_combo = QComboBox()
        self._tp_combo.addItem("— (material-only)", None)
        for i in range(TARGET_POSITION_MIN, TARGET_POSITION_MAX + 1):
            self._tp_combo.addItem(str(i), i)
        self._notes = QLineEdit()
        self._priority = QSpinBox()
        self._priority.setRange(-1000, 10000)
        self._lo_min = QLineEdit()
        self._lo_max = QLineEdit()
        self._lo_step = QLineEdit()
        self._rules_table = QTableWidget(0, 4)
        self._rules_table.setHorizontalHeaderLabels(
            ["Driver min", "Driver max", "IA min", "IA max"]
        )
        self._rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._rules_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._rules_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._rules_table.setMinimumHeight(120)

        form = QFormLayout()
        form.addRow("Id:", self._id_edit)
        form.addRow(self._enabled)
        form.addRow("Material:", self._material)
        form.addRow("Target position:", self._tp_combo)
        form.addRow("Notes:", self._notes)
        form.addRow("Priority:", self._priority)
        self._lo_box = QGroupBox("linear_offset bounds (legacy projection)")
        lo_l = QFormLayout()
        lo_l.addRow("Min:", self._lo_min)
        lo_l.addRow("Max:", self._lo_max)
        lo_l.addRow("Step:", self._lo_step)
        self._lo_box.setLayout(lo_l)
        form.addRow(self._lo_box)
        rt_label = QLabel("incident_angle rules (driver = linear_offset)")
        form.addRow(rt_label)
        form.addRow(self._rules_table)
        rbtn = QHBoxLayout()
        self._rule_add = QPushButton("Add rule row")
        self._rule_del = QPushButton("Remove selected")
        rbtn.addWidget(self._rule_add)
        rbtn.addWidget(self._rule_del)
        form.addRow(rbtn)

        self._rule_add.clicked.connect(self._add_rule_row)
        self._rule_del.clicked.connect(self._remove_rule_rows)
        self._rules_table.itemChanged.connect(lambda *_: self._commit_row(self._list.currentRow()))

        for w in (
            self._id_edit,
            self._material,
            self._notes,
            self._lo_min,
            self._lo_max,
            self._lo_step,
        ):
            w.editingFinished.connect(lambda: self._commit_row(self._list.currentRow()))
        self._enabled.stateChanged.connect(lambda _: self._commit_row(self._list.currentRow()))
        self._tp_combo.currentIndexChanged.connect(lambda _: self._commit_row(self._list.currentRow()))
        self._priority.valueChanged.connect(lambda _: self._commit_row(self._list.currentRow()))

        right = QWidget()
        right.setLayout(form)
        scroll_right = QScrollArea()
        scroll_right.setWidgetResizable(True)
        scroll_right.setFrameShape(QFrame.Shape.NoFrame)
        scroll_right.setWidget(right)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._list)
        split.addWidget(scroll_right)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add profile")
        self._btn_dup = QPushButton("Duplicate")
        self._btn_del = QPushButton("Delete")
        for b in (self._btn_add, self._btn_dup, self._btn_del):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        self._btn_add.clicked.connect(self._add_profile)
        self._btn_dup.clicked.connect(self._dup_profile)
        self._btn_del.clicked.connect(self._del_profile)

        help_txt = QLabel(
            "<b>How to use</b>: Pick a profile in the list (left). "
            "<b>preset-tp-1 … preset-tp-12</b> are fixed constants from Prompt 2.1 (linear_offset + incident_angle vs "
            "linear_offset); they are read-only except <b>Enabled</b>. "
            "Use <b>Add profile</b> for custom material/target rules. "
            "<b>Precedence</b>: global registry → these presets → material-only → material+target."
        )
        help_txt.setWordWrap(True)
        outer = QVBoxLayout()
        outer.addWidget(help_txt)
        outer.addLayout(btn_row)
        outer.addWidget(split)
        self.setLayout(outer)

    def _apply_preset_form_readonly(self, locked: bool) -> None:
        """Shipped preset-tp-* rows are constants (Prompt 2.1); only Enabled may be toggled."""
        self._id_edit.setReadOnly(locked)
        self._material.setReadOnly(locked)
        self._tp_combo.setEnabled(not locked)
        self._notes.setReadOnly(locked)
        self._priority.setEnabled(not locked)
        self._lo_min.setReadOnly(locked)
        self._lo_max.setReadOnly(locked)
        self._lo_step.setReadOnly(locked)
        self._lo_box.setEnabled(not locked)
        self._rules_table.setEnabled(not locked)
        self._rule_add.setEnabled(not locked)
        self._rule_del.setEnabled(not locked)
        self._btn_del.setEnabled(not locked)
        self._enabled.setEnabled(True)

    def set_profiles(self, profiles: list[dict[str, Any]]) -> None:
        self._building = True
        self._profiles = [dict(p) for p in profiles if isinstance(p, dict)]
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._profiles:
            self._list.addItem(QListWidgetItem(self._label_for(p)))
        if self._profiles:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._building = False
        self._prev_row = -1
        if self._profiles:
            self._load_row(0)
            self._prev_row = 0
        else:
            self._clear_form()

    def get_profiles(self) -> list[dict[str, Any]]:
        r = self._list.currentRow()
        if r >= 0:
            self._commit_row(r)
        return [deepcopy_dict(p) for p in self._profiles]

    def validate_all(self) -> list[str]:
        r = self._list.currentRow()
        if r >= 0:
            self._commit_row(r)
        errs: list[str] = []
        for p in self._profiles:
            errs.extend(validate_scoped_parameter_profile(p))
        return errs

    @staticmethod
    def _label_for(p: dict[str, Any]) -> str:
        tid = p.get("id", "?")
        mat = p.get("material_name", "") or ""
        tp = p.get("target_position")
        tp_s = str(tp) if tp is not None and str(tp).strip() != "" else "—"
        return f"{tid} | {mat!r} | tp={tp_s}"

    def _clear_form(self) -> None:
        self._id_edit.clear()
        self._enabled.setChecked(True)
        self._material.clear()
        self._tp_combo.setCurrentIndex(0)
        self._notes.clear()
        self._priority.setValue(0)
        self._lo_min.clear()
        self._lo_max.clear()
        self._lo_step.clear()
        self._rules_table.setRowCount(0)
        self._apply_preset_form_readonly(False)

    def _load_row(self, row: int) -> None:
        self._building = True
        if row < 0 or row >= len(self._profiles):
            self._clear_form()
            self._building = False
            return
        p = self._profiles[row]
        self._id_edit.setText(str(p.get("id", "")))
        self._enabled.setChecked(bool(p.get("enabled", True)))
        self._material.setText(str(p.get("material_name", "") or ""))
        tp = p.get("target_position")
        idx = 0
        if tp is not None and str(tp).strip() != "":
            try:
                tpi = int(tp)
            except (TypeError, ValueError):
                tpi = None
            if tpi is not None:
                for i in range(self._tp_combo.count()):
                    d = self._tp_combo.itemData(i)
                    if d is not None and int(d) == tpi:
                        idx = i
                        break
        self._tp_combo.setCurrentIndex(idx)
        self._notes.setText(str(p.get("notes", "") or ""))
        self._priority.setValue(int(p.get("priority", 0) or 0))

        po = p.get("parameter_overrides") or {}
        lo = po.get("linear_offset") if isinstance(po, dict) else None
        if isinstance(lo, dict):
            self._lo_min.setText("" if lo.get("min_value") is None else str(lo.get("min_value")))
            self._lo_max.setText("" if lo.get("max_value") is None else str(lo.get("max_value")))
            st = lo.get("step")
            self._lo_step.setText("" if st is None else str(st))
        else:
            self._lo_min.clear()
            self._lo_max.clear()
            self._lo_step.clear()

        self._rules_table.setRowCount(0)
        ia = po.get("incident_angle") if isinstance(po, dict) else None
        dcs = ia.get("dependent_constraints") if isinstance(ia, dict) else None
        rules: list[dict[str, Any]] = []
        if isinstance(dcs, list) and dcs:
            dc0 = dcs[0]
            if isinstance(dc0, dict):
                rlist = dc0.get("rules")
                if isinstance(rlist, list):
                    rules = [r for r in rlist if isinstance(r, dict)]
        for r in rules:
            rr = self._rules_table.rowCount()
            self._rules_table.insertRow(rr)
            for c, key in enumerate(("driver_min", "driver_max", "min_value", "max_value")):
                v = r.get(key)
                self._rules_table.setItem(rr, c, QTableWidgetItem("" if v is None else str(v)))
        self._apply_preset_form_readonly(is_shipped_target_position_preset(p))
        self._building = False

    def _commit_row(self, row: int) -> None:
        if self._building:
            return
        if row < 0 or row >= len(self._profiles):
            return
        p = self._profiles[row]
        if is_shipped_target_position_preset(p):
            p["enabled"] = self._enabled.isChecked()
            item = self._list.item(row)
            if item:
                item.setText(self._label_for(p))
            return
        p["id"] = self._id_edit.text().strip() or p.get("id", "profile")
        p["enabled"] = self._enabled.isChecked()
        p["material_name"] = self._material.text().strip()
        tp = self._tp_combo.currentData()
        if tp is None:
            p.pop("target_position", None)
        else:
            p["target_position"] = int(tp)
        p["notes"] = self._notes.text().strip()
        p["priority"] = int(self._priority.value())

        lo: dict[str, Any] = {}
        a, b, st = self._lo_min.text(), self._lo_max.text(), self._lo_step.text()
        if a.strip() or b.strip() or st.strip():
            lo = {"type": "continuous", "is_enabled": True}
            if a.strip():
                lo["min_value"] = _safe_float(a)
            if b.strip():
                lo["max_value"] = _safe_float(b)
            if st.strip():
                lo["step"] = float(_safe_float(st) or 0.0)
            else:
                lo["step"] = 0.0

        rules = []
        for ri in range(self._rules_table.rowCount()):
            row_items = [self._rules_table.item(ri, c) for c in range(4)]
            raw = [(it.text() if it else "").strip() for it in row_items]
            if not any(raw):
                continue
            rules.append(
                {
                    "driver_min": _safe_float(raw[0]),
                    "driver_max": _safe_float(raw[1]),
                    "min_value": _safe_float(raw[2]),
                    "max_value": _safe_float(raw[3]),
                }
            )

        po: dict[str, Any] = {}
        if lo:
            po["linear_offset"] = lo
        if rules:
            po["incident_angle"] = {"dependent_constraints": [{"driver_parameter": "linear_offset", "rules": rules}]}
        p["parameter_overrides"] = po

        item = self._list.item(row)
        if item:
            item.setText(self._label_for(p))

    def _commit_current(self) -> None:
        self._commit_row(self._list.currentRow())

    def _on_row_changed(self, row: int) -> None:
        if self._building:
            return
        prev = self._prev_row
        if prev >= 0 and prev < len(self._profiles) and prev != row:
            self._commit_row(prev)
        if row >= 0:
            self._load_row(row)
        else:
            self._clear_form()
        self._prev_row = row

    def _add_profile(self) -> None:
        cr = self._list.currentRow()
        if cr >= 0:
            self._commit_row(cr)
        n = len(self._profiles) + 1
        new_p = {
            "id": f"profile-{n}",
            "enabled": True,
            "material_name": "",
            "target_position": 1,
            "notes": "",
            "priority": 10,
            "parameter_overrides": {},
        }
        self._profiles.append(new_p)
        self._list.addItem(QListWidgetItem(self._label_for(new_p)))
        self._list.setCurrentRow(len(self._profiles) - 1)

    def _dup_profile(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._commit_row(row)
        dup = deepcopy(self._profiles[row])
        dup["id"] = str(dup.get("id", "profile")) + "-copy"
        self._profiles.append(dup)
        self._list.addItem(QListWidgetItem(self._label_for(dup)))
        self._list.setCurrentRow(len(self._profiles) - 1)

    def _del_profile(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        if is_shipped_target_position_preset(self._profiles[row]):
            return
        self._profiles.pop(row)
        self._list.takeItem(row)
        self._prev_row = -1
        if self._profiles:
            nr = min(row, len(self._profiles) - 1)
            self._list.setCurrentRow(nr)
            self._prev_row = nr
        else:
            self._clear_form()

    def _add_rule_row(self) -> None:
        r = self._rules_table.rowCount()
        self._rules_table.insertRow(r)
        for c in range(4):
            self._rules_table.setItem(r, c, QTableWidgetItem(""))

    def _remove_rule_rows(self) -> None:
        rows = sorted({i.row() for i in self._rules_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._rules_table.removeRow(r)


def deepcopy_dict(d: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(d)


class ScopedSpecProfilesEditor(QWidget):
    """List + form for ``spec_profiles`` (D.Spec + optional max_lifetime)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._profiles: list[dict[str, Any]] = []
        self._building = False
        self._prev_row = -1

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)

        self._id_edit = QLineEdit()
        self._enabled = QCheckBox("Enabled")
        self._material = QLineEdit()
        self._tp_combo = QComboBox()
        self._tp_combo.addItem("— (material-only)", None)
        for i in range(TARGET_POSITION_MIN, TARGET_POSITION_MAX + 1):
            self._tp_combo.addItem(str(i), i)
        self._notes = QLineEdit()
        self._priority = QSpinBox()
        self._priority.setRange(-1000, 10000)
        self._max_lt = QLineEdit()
        self._max_lt.setPlaceholderText("optional max lifetime")

        # Spec numeric fields (empty = unset)
        self._rs_t = QLineEdit()
        self._rs_tol = QLineEdit()
        self._rs_min = QLineEdit()
        self._rs_max = QLineEdit()
        self._th_t = QLineEdit()
        self._th_tol = QLineEdit()
        self._th_min = QLineEdit()
        self._th_max = QLineEdit()
        self._rsu = QLineEdit()
        self._use_rs = QCheckBox("use_rs_spec")
        self._use_th = QCheckBox("use_thickness_spec")
        self._use_rsu = QCheckBox("use_rsu_spec")
        self._use_rs_tm = QCheckBox("use_rs_target_mode")
        self._use_th_tm = QCheckBox("use_thickness_target_mode")

        form = QFormLayout()
        form.addRow("Id:", self._id_edit)
        form.addRow(self._enabled)
        form.addRow("Material:", self._material)
        form.addRow("Target position:", self._tp_combo)
        form.addRow("Notes:", self._notes)
        form.addRow("Priority:", self._priority)
        form.addRow("Max lifetime (optional):", self._max_lt)
        sg = QGroupBox("Spec fields")
        fl = QFormLayout()
        fl.addRow("rs_target:", self._rs_t)
        fl.addRow("rs_tol:", self._rs_tol)
        fl.addRow("rs_min:", self._rs_min)
        fl.addRow("rs_max:", self._rs_max)
        fl.addRow("thickness_target:", self._th_t)
        fl.addRow("thickness_tol:", self._th_tol)
        fl.addRow("thickness_min:", self._th_min)
        fl.addRow("thickness_max:", self._th_max)
        fl.addRow("rsu_max:", self._rsu)
        fl.addRow(self._use_rs)
        fl.addRow(self._use_th)
        fl.addRow(self._use_rsu)
        fl.addRow(self._use_rs_tm)
        fl.addRow(self._use_th_tm)
        sg.setLayout(fl)
        form.addRow(sg)

        for w in (
            self._id_edit,
            self._material,
            self._notes,
            self._max_lt,
            self._rs_t,
            self._rs_tol,
            self._rs_min,
            self._rs_max,
            self._th_t,
            self._th_tol,
            self._th_min,
            self._th_max,
            self._rsu,
        ):
            w.editingFinished.connect(lambda: self._commit_row(self._list.currentRow()))
        self._enabled.stateChanged.connect(lambda _: self._commit_row(self._list.currentRow()))
        self._tp_combo.currentIndexChanged.connect(lambda _: self._commit_row(self._list.currentRow()))
        self._priority.valueChanged.connect(lambda _: self._commit_row(self._list.currentRow()))
        for cb in (
            self._use_rs,
            self._use_th,
            self._use_rsu,
            self._use_rs_tm,
            self._use_th_tm,
        ):
            cb.stateChanged.connect(lambda _: self._commit_row(self._list.currentRow()))

        right = QWidget()
        right.setLayout(form)
        scroll_right = QScrollArea()
        scroll_right.setWidgetResizable(True)
        scroll_right.setFrameShape(QFrame.Shape.NoFrame)
        scroll_right.setWidget(right)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._list)
        split.addWidget(scroll_right)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add profile")
        self._btn_dup = QPushButton("Duplicate")
        self._btn_del = QPushButton("Delete")
        for b in (self._btn_add, self._btn_dup, self._btn_del):
            btn_row.addWidget(b)
        self._btn_add.clicked.connect(self._add_profile)
        self._btn_dup.clicked.connect(self._dup_profile)
        self._btn_del.clicked.connect(self._del_profile)

        spec_help = QLabel(
            "<b>How to use</b>: Optional RS/thickness/RSU spec thresholds per material and/or target position. "
            "Merged after global <code>spec_settings</code> and legacy <code>material_overrides</code>. "
            "Leave fields empty to inherit; use <b>max lifetime</b> when needed."
        )
        spec_help.setWordWrap(True)
        outer = QVBoxLayout()
        outer.addWidget(spec_help)
        outer.addLayout(btn_row)
        outer.addWidget(split)
        self.setLayout(outer)

    def set_profiles(self, profiles: list[dict[str, Any]]) -> None:
        self._building = True
        self._profiles = [dict(p) for p in profiles if isinstance(p, dict)]
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._profiles:
            self._list.addItem(QListWidgetItem(ScopedParameterProfilesEditor._label_for(p)))
        if self._profiles:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._building = False
        self._prev_row = -1
        if self._profiles:
            self._load_row(0)
            self._prev_row = 0
        else:
            self._clear_form()

    def get_profiles(self) -> list[dict[str, Any]]:
        r = self._list.currentRow()
        if r >= 0:
            self._commit_row(r)
        return [deepcopy_dict(p) for p in self._profiles]

    def validate_all(self) -> list[str]:
        r = self._list.currentRow()
        if r >= 0:
            self._commit_row(r)
        errs: list[str] = []
        for p in self._profiles:
            errs.extend(validate_scoped_spec_profile(p))
        return errs

    def _clear_form(self) -> None:
        self._id_edit.clear()
        self._enabled.setChecked(True)
        self._material.clear()
        self._tp_combo.setCurrentIndex(0)
        self._notes.clear()
        self._priority.setValue(0)
        self._max_lt.clear()
        for w in (
            self._rs_t,
            self._rs_tol,
            self._rs_min,
            self._rs_max,
            self._th_t,
            self._th_tol,
            self._th_min,
            self._th_max,
            self._rsu,
        ):
            w.clear()
        self._use_rs.setChecked(True)
        self._use_th.setChecked(True)
        self._use_rsu.setChecked(False)
        self._use_rs_tm.setChecked(False)
        self._use_th_tm.setChecked(False)

    def _load_row(self, row: int) -> None:
        self._building = True
        if row < 0 or row >= len(self._profiles):
            self._clear_form()
            self._building = False
            return
        p = self._profiles[row]
        self._id_edit.setText(str(p.get("id", "")))
        self._enabled.setChecked(bool(p.get("enabled", True)))
        self._material.setText(str(p.get("material_name", "") or ""))
        tp = p.get("target_position")
        idx = 0
        if tp is not None and str(tp).strip() != "":
            try:
                tpi = int(tp)
            except (TypeError, ValueError):
                tpi = None
            if tpi is not None:
                for i in range(self._tp_combo.count()):
                    d = self._tp_combo.itemData(i)
                    if d is not None and int(d) == tpi:
                        idx = i
                        break
        self._tp_combo.setCurrentIndex(idx)
        self._notes.setText(str(p.get("notes", "") or ""))
        self._priority.setValue(int(p.get("priority", 0) or 0))
        ml = p.get("max_lifetime")
        self._max_lt.setText("" if ml is None else str(ml))

        spec = p.get("spec") if isinstance(p.get("spec"), dict) else {}

        def _set(le: QLineEdit, key: str) -> None:
            v = spec.get(key)
            le.setText("" if v is None else str(v))

        _set(self._rs_t, "rs_target")
        _set(self._rs_tol, "rs_tol")
        _set(self._rs_min, "rs_min")
        _set(self._rs_max, "rs_max")
        _set(self._th_t, "thickness_target")
        _set(self._th_tol, "thickness_tol")
        _set(self._th_min, "thickness_min")
        _set(self._th_max, "thickness_max")
        _set(self._rsu, "rsu_max")

        self._use_rs.setChecked(bool(spec.get("use_rs_spec", True)))
        self._use_th.setChecked(bool(spec.get("use_thickness_spec", True)))
        self._use_rsu.setChecked(bool(spec.get("use_rsu_spec", False)))
        self._use_rs_tm.setChecked(bool(spec.get("use_rs_target_mode", False)))
        self._use_th_tm.setChecked(bool(spec.get("use_thickness_target_mode", False)))
        self._building = False

    def _commit_row(self, row: int) -> None:
        if self._building:
            return
        if row < 0 or row >= len(self._profiles):
            return
        p = self._profiles[row]
        p["id"] = self._id_edit.text().strip() or p.get("id", "spec-profile")
        p["enabled"] = self._enabled.isChecked()
        p["material_name"] = self._material.text().strip()
        tp = self._tp_combo.currentData()
        if tp is None:
            p.pop("target_position", None)
        else:
            p["target_position"] = int(tp)
        p["notes"] = self._notes.text().strip()
        p["priority"] = int(self._priority.value())
        ml = self._max_lt.text().strip()
        if ml:
            try:
                p["max_lifetime"] = float(ml)
            except ValueError:
                p["max_lifetime"] = None
        else:
            p.pop("max_lifetime", None)

        spec: dict[str, Any] = {}
        pairs = [
            ("rs_target", self._rs_t),
            ("rs_tol", self._rs_tol),
            ("rs_min", self._rs_min),
            ("rs_max", self._rs_max),
            ("thickness_target", self._th_t),
            ("thickness_tol", self._th_tol),
            ("thickness_min", self._th_min),
            ("thickness_max", self._th_max),
            ("rsu_max", self._rsu),
        ]
        for key, ed in pairs:
            v = _safe_float(ed.text())
            if v is not None:
                spec[key] = v
        spec["use_rs_spec"] = self._use_rs.isChecked()
        spec["use_thickness_spec"] = self._use_th.isChecked()
        spec["use_rsu_spec"] = self._use_rsu.isChecked()
        spec["use_rs_target_mode"] = self._use_rs_tm.isChecked()
        spec["use_thickness_target_mode"] = self._use_th_tm.isChecked()
        p["spec"] = spec

        item = self._list.item(row)
        if item:
            item.setText(ScopedParameterProfilesEditor._label_for(p))

    def _commit_current(self) -> None:
        self._commit_row(self._list.currentRow())

    def _on_row_changed(self, row: int) -> None:
        if self._building:
            return
        prev = self._prev_row
        if prev >= 0 and prev < len(self._profiles) and prev != row:
            self._commit_row(prev)
        if row >= 0:
            self._load_row(row)
        else:
            self._clear_form()
        self._prev_row = row

    def _add_profile(self) -> None:
        cr = self._list.currentRow()
        if cr >= 0:
            self._commit_row(cr)
        n = len(self._profiles) + 1
        new_p = {
            "id": f"spec-profile-{n}",
            "enabled": True,
            "material_name": "",
            "notes": "",
            "priority": 10,
            "spec": {},
        }
        self._profiles.append(new_p)
        self._list.addItem(QListWidgetItem(ScopedParameterProfilesEditor._label_for(new_p)))
        self._list.setCurrentRow(len(self._profiles) - 1)

    def _dup_profile(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._commit_row(row)
        dup = deepcopy(self._profiles[row])
        dup["id"] = str(dup.get("id", "spec")) + "-copy"
        self._profiles.append(dup)
        self._list.addItem(QListWidgetItem(ScopedParameterProfilesEditor._label_for(dup)))
        self._list.setCurrentRow(len(self._profiles) - 1)

    def _del_profile(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._profiles.pop(row)
        self._list.takeItem(row)
        self._prev_row = -1
        if self._profiles:
            nr = min(row, len(self._profiles) - 1)
            self._list.setCurrentRow(nr)
            self._prev_row = nr
        else:
            self._clear_form()
