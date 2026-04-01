"""
Structured parameter definitions, alias lookup, range constraints, and validation.

Runtime code still consumes ``parameter_constraints`` dicts (legacy shape). Use
``apply_parameter_registry_to_config`` after loading effective config to merge
registry files + legacy settings into ``config["parameter_constraints"]``.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.config_store import (
    deep_merge,
    get_default_parameter_registry_path,
    get_user_parameter_registry_path,
    load_json,
    save_json,
)


class ParameterDataType(str, Enum):
    """Logical process-parameter kind (maps to legacy ``type`` string where needed)."""

    CONTINUOUS = "continuous"
    DISCRETE_STEP = "discrete_step"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterAlias:
    """Maps an Excel header (or other label) to a canonical parameter."""

    header: str
    source: str = "excel"


@dataclass
class SimpleRangeConstraint:
    """Static min/max/step bounds for a parameter."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


@dataclass
class DependentRangeRule:
    """When ``driver_parameter`` is within [driver_min, driver_max], apply bounds."""

    driver_min: Optional[float] = None
    driver_max: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class DependentRangeConstraint:
    """Bounds that depend on another parameter's value (by canonical name)."""

    driver_parameter: str
    rules: list[DependentRangeRule] = field(default_factory=list)


@dataclass
class ValidationIssue:
    code: str
    message: str
    field: Optional[str] = None


@dataclass
class ParameterValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class ParameterDefinition:
    canonical_name: str
    display_name: str
    data_type: ParameterDataType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: float = 0.0
    enabled: bool = True
    notes: str = ""
    unit: Optional[str] = None
    aliases: list[ParameterAlias] = field(default_factory=list)
    allowed_values: Optional[list[Any]] = None
    is_coupled: bool = False
    coupling_group: Optional[str] = None
    simple_range: Optional[SimpleRangeConstraint] = None
    dependent_constraints: list[DependentRangeConstraint] = field(default_factory=list)

    def effective_min(self) -> Optional[float]:
        if self.simple_range is not None and self.simple_range.min_value is not None:
            return self.simple_range.min_value
        return self.min_value

    def effective_max(self) -> Optional[float]:
        if self.simple_range is not None and self.simple_range.max_value is not None:
            return self.simple_range.max_value
        return self.max_value

    def effective_step(self) -> float:
        if self.simple_range is not None and self.simple_range.step is not None:
            return float(self.simple_range.step)
        return float(self.step or 0.0)


class ParameterRegistry:
    """Lookup by canonical name, display name, or any registered alias (e.g. Excel header)."""

    def __init__(self) -> None:
        self._by_canonical: dict[str, ParameterDefinition] = {}
        self._display_lower: dict[str, str] = {}
        self._alias_lower: dict[str, str] = {}

    @property
    def parameters(self) -> dict[str, ParameterDefinition]:
        return dict(self._by_canonical)

    def add(self, definition: ParameterDefinition) -> None:
        c = definition.canonical_name
        self._by_canonical[c] = definition
        self._display_lower[definition.display_name.strip().lower()] = c
        for a in definition.aliases:
            self._alias_lower[a.header.strip().lower()] = c

    def rebuild_indexes(self) -> None:
        self._display_lower.clear()
        self._alias_lower.clear()
        for d in self._by_canonical.values():
            self._display_lower[d.display_name.strip().lower()] = d.canonical_name
            for a in d.aliases:
                self._alias_lower[a.header.strip().lower()] = d.canonical_name

    def get_by_canonical(self, name: str) -> Optional[ParameterDefinition]:
        return self._by_canonical.get(name)

    def get_by_display_name(self, display: str) -> Optional[ParameterDefinition]:
        c = self._display_lower.get(display.strip().lower())
        return self._by_canonical.get(c) if c else None

    def get_by_alias_header(self, header: str) -> Optional[ParameterDefinition]:
        c = self._alias_lower.get(header.strip().lower())
        return self._by_canonical.get(c) if c else None

    def resolve_header(self, header: str) -> Optional[ParameterDefinition]:
        """Resolve Excel column title: try alias first, then display name, then canonical."""
        h = header.strip()
        by_alias = self.get_by_alias_header(h)
        if by_alias:
            return by_alias
        by_disp = self.get_by_display_name(h)
        if by_disp:
            return by_disp
        return self.get_by_canonical(h)

    def to_legacy_parameter_constraints(self) -> dict[str, Any]:
        """Shape expected by ``candidate_generator``, ``constraints``, and optimizers."""
        out: dict[str, Any] = {}
        for name, d in self._by_canonical.items():
            out[name] = parameter_definition_to_legacy_entry(d)
        return out


def _legacy_type_to_data_type(raw: Any) -> ParameterDataType:
    t = str(raw or "continuous").lower()
    if t == "integer":
        return ParameterDataType.DISCRETE_STEP
    if t == "boolean":
        return ParameterDataType.BOOLEAN
    if t in ("categorical", "category"):
        return ParameterDataType.CATEGORICAL
    return ParameterDataType.CONTINUOUS


def _data_type_to_legacy_type(dt: ParameterDataType, step: float) -> str:
    if dt == ParameterDataType.BOOLEAN:
        return "boolean"
    if dt == ParameterDataType.CATEGORICAL:
        return "continuous"
    if dt == ParameterDataType.DISCRETE_STEP:
        return "integer" if float(step or 0.0) == 1.0 else "continuous"
    return "continuous"


def parameter_definition_to_legacy_entry(d: ParameterDefinition) -> dict[str, Any]:
    st = d.effective_step()
    return {
        "type": _data_type_to_legacy_type(d.data_type, st),
        "min_value": d.effective_min(),
        "max_value": d.effective_max(),
        "step": st,
        "allowed_values": d.allowed_values,
        "is_enabled": d.enabled,
        "is_coupled": d.is_coupled,
        "coupling_group": d.coupling_group,
    }


def _parse_simple_range(obj: Any) -> Optional[SimpleRangeConstraint]:
    if not isinstance(obj, dict):
        return None
    return SimpleRangeConstraint(
        min_value=_maybe_float(obj.get("min_value")),
        max_value=_maybe_float(obj.get("max_value")),
        step=_maybe_float(obj.get("step")),
    )


def _parse_dependent_constraints(items: Any) -> list[DependentRangeConstraint]:
    out: list[DependentRangeConstraint] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        rules_raw = it.get("rules", [])
        rules: list[DependentRangeRule] = []
        if isinstance(rules_raw, list):
            for r in rules_raw:
                if not isinstance(r, dict):
                    continue
                rules.append(
                    DependentRangeRule(
                        driver_min=_maybe_float(r.get("driver_min")),
                        driver_max=_maybe_float(r.get("driver_max")),
                        min_value=_maybe_float(r.get("min_value")),
                        max_value=_maybe_float(r.get("max_value")),
                    )
                )
        out.append(
            DependentRangeConstraint(driver_parameter=str(it.get("driver_parameter", "")), rules=rules)
        )
    return out


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_aliases(items: Any) -> list[ParameterAlias]:
    out: list[ParameterAlias] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, str):
            out.append(ParameterAlias(header=it))
        elif isinstance(it, dict):
            h = it.get("header")
            if h is not None:
                out.append(ParameterAlias(header=str(h), source=str(it.get("source", "excel"))))
    return out


def _parse_data_type(raw: Any) -> ParameterDataType:
    if isinstance(raw, ParameterDataType):
        return raw
    s = str(raw or "continuous").lower()
    if s in ("integer", "discrete_step"):
        return ParameterDataType.DISCRETE_STEP
    if s in ("boolean", "bool"):
        return ParameterDataType.BOOLEAN
    if s in ("categorical", "category"):
        return ParameterDataType.CATEGORICAL
    if s == "continuous":
        return ParameterDataType.CONTINUOUS
    try:
        return ParameterDataType(s)
    except ValueError:
        return ParameterDataType.CONTINUOUS


def parameter_definition_from_dict(obj: dict[str, Any]) -> ParameterDefinition:
    dt_raw = obj.get("data_type", obj.get("type"))
    data_type = _parse_data_type(dt_raw)

    sr = _parse_simple_range(obj.get("simple_range"))
    return ParameterDefinition(
        canonical_name=str(obj["canonical_name"]),
        display_name=str(obj.get("display_name") or obj["canonical_name"]),
        data_type=data_type,
        min_value=_maybe_float(obj.get("min_value")),
        max_value=_maybe_float(obj.get("max_value")),
        step=float(obj.get("step", 0.0) or 0.0),
        enabled=bool(obj.get("enabled", obj.get("is_enabled", True))),
        notes=str(obj.get("notes", "") or ""),
        unit=obj.get("unit"),
        aliases=_parse_aliases(obj.get("aliases", [])),
        allowed_values=obj.get("allowed_values"),
        is_coupled=bool(obj.get("is_coupled", False)),
        coupling_group=obj.get("coupling_group"),
        simple_range=sr,
        dependent_constraints=_parse_dependent_constraints(obj.get("dependent_constraints", [])),
    )


def registry_from_dict(data: dict[str, Any]) -> ParameterRegistry:
    reg = ParameterRegistry()
    params = data.get("parameters", [])
    if isinstance(params, list):
        for p in params:
            if isinstance(p, dict):
                reg.add(parameter_definition_from_dict(p))
    return reg


def registry_to_dict(reg: ParameterRegistry) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for d in reg._by_canonical.values():
        rows.append(parameter_definition_to_dict(d))
    return {"version": 1, "parameters": rows}


def parameter_definition_to_dict(d: ParameterDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {
        "canonical_name": d.canonical_name,
        "display_name": d.display_name,
        "data_type": d.data_type.value,
        "min_value": d.min_value,
        "max_value": d.max_value,
        "step": d.step,
        "enabled": d.enabled,
        "notes": d.notes,
        "unit": d.unit,
        "aliases": [{"header": a.header, "source": a.source} for a in d.aliases],
        "allowed_values": d.allowed_values,
        "is_coupled": d.is_coupled,
        "coupling_group": d.coupling_group,
        "dependent_constraints": [
            {
                "driver_parameter": dc.driver_parameter,
                "rules": [
                    {
                        "driver_min": r.driver_min,
                        "driver_max": r.driver_max,
                        "min_value": r.min_value,
                        "max_value": r.max_value,
                    }
                    for r in dc.rules
                ],
            }
            for dc in d.dependent_constraints
        ],
    }
    if d.simple_range is not None:
        out["simple_range"] = {
            "min_value": d.simple_range.min_value,
            "max_value": d.simple_range.max_value,
            "step": d.simple_range.step,
        }
    return out


def load_parameter_registry_file(path: Path) -> ParameterRegistry:
    raw = load_json(path)
    if not raw:
        return ParameterRegistry()
    return registry_from_dict(raw)


def merge_registries(base: ParameterRegistry, override: ParameterRegistry) -> ParameterRegistry:
    merged = ParameterRegistry()
    for d in base._by_canonical.values():
        merged.add(deepcopy(d))
    for name, d in override._by_canonical.items():
        merged._by_canonical[name] = deepcopy(d)
    merged.rebuild_indexes()
    return merged


def load_effective_parameter_registry(project_root: Path) -> ParameterRegistry:
    default_path = get_default_parameter_registry_path(project_root)
    user_path = get_user_parameter_registry_path(project_root)
    base = load_parameter_registry_file(default_path)
    user = load_parameter_registry_file(user_path)
    return merge_registries(base, user)


def save_user_parameter_registry(project_root: Path, reg: ParameterRegistry) -> None:
    path = get_user_parameter_registry_path(project_root)
    save_json(path, registry_to_dict(reg))


def migrate_legacy_parameter_constraints(
    legacy: dict[str, Any],
    column_mapping: Optional[dict[str, Any]] = None,
) -> ParameterRegistry:
    """
    Build a registry from legacy ``parameter_constraints`` and optional ``column_mapping``.

    Aliases are inferred by reversing column_mapping (internal field -> Excel header).
    """
    reg = ParameterRegistry()
    reverse_map: dict[str, str] = {}
    if column_mapping:
        for excel_header, internal in column_mapping.items():
            if isinstance(internal, str):
                reverse_map[internal] = str(excel_header)

    for pname, cfg in (legacy or {}).items():
        if not isinstance(cfg, dict):
            continue
        dt = _legacy_type_to_data_type(cfg.get("type"))
        display = reverse_map.get(pname, pname.replace("_", " ").title())
        aliases: list[ParameterAlias] = []
        eh = reverse_map.get(pname)
        if eh:
            aliases.append(ParameterAlias(header=eh))

        reg.add(
            ParameterDefinition(
                canonical_name=str(pname),
                display_name=display,
                data_type=dt,
                min_value=_maybe_float(cfg.get("min_value")),
                max_value=_maybe_float(cfg.get("max_value")),
                step=float(cfg.get("step", 0.0) or 0.0),
                enabled=bool(cfg.get("is_enabled", True)),
                aliases=aliases,
                allowed_values=cfg.get("allowed_values"),
                is_coupled=bool(cfg.get("is_coupled", False)),
                coupling_group=cfg.get("coupling_group"),
            )
        )
    return reg


def _apply_legacy_dict_to_definition(defn: ParameterDefinition, cfg: dict[str, Any]) -> None:
    """
    Apply a legacy ``parameter_constraints`` entry onto *defn* (mutates in place).

    Clears ``simple_range`` so min/max/step match the flat fields used by the legacy
    projection (same as :func:`parameter_definition_to_legacy_entry`).
    """
    defn.data_type = _legacy_type_to_data_type(cfg.get("type"))
    defn.min_value = _maybe_float(cfg.get("min_value"))
    defn.max_value = _maybe_float(cfg.get("max_value"))
    defn.step = float(cfg.get("step", 0.0) or 0.0)
    defn.enabled = bool(cfg.get("is_enabled", True))
    defn.allowed_values = cfg.get("allowed_values")
    defn.is_coupled = bool(cfg.get("is_coupled", False))
    defn.coupling_group = cfg.get("coupling_group")
    defn.simple_range = None


def build_effective_registry(
    base_reg: ParameterRegistry,
    merged_pc: dict[str, Any],
    column_mapping: Optional[dict[str, Any]] = None,
) -> ParameterRegistry:
    """
    Copy *base_reg* and sync every parameter from *merged_pc* (final legacy dict).

    Parameters present only in *merged_pc* are added using the same rules as
    :func:`migrate_legacy_parameter_constraints` (for display names / aliases).
    """
    out = ParameterRegistry()
    for d in base_reg._by_canonical.values():
        out.add(deepcopy(d))

    for pname, pcfg in (merged_pc or {}).items():
        if not isinstance(pcfg, dict):
            continue
        existing = out.get_by_canonical(pname)
        if existing is not None:
            _apply_legacy_dict_to_definition(existing, pcfg)
        else:
            sub = migrate_legacy_parameter_constraints({pname: pcfg}, column_mapping)
            for d in sub._by_canonical.values():
                out.add(deepcopy(d))
    out.rebuild_indexes()
    return out


def validate_parameter_definition(defn: ParameterDefinition) -> ParameterValidationResult:
    issues: list[ValidationIssue] = []
    if not defn.canonical_name.strip():
        issues.append(ValidationIssue("canonical_name_empty", "canonical_name must be non-empty", "canonical_name"))
    if not defn.display_name.strip():
        issues.append(ValidationIssue("display_name_empty", "display_name must be non-empty", "display_name"))

    emin, emax = defn.effective_min(), defn.effective_max()
    if emin is not None and emax is not None and emin > emax + 1e-12:
        issues.append(ValidationIssue("min_gt_max", "min_value is greater than max_value", "min_value"))

    st = defn.effective_step()
    if st < 0:
        issues.append(ValidationIssue("step_negative", "step must be non-negative", "step"))

    if defn.data_type == ParameterDataType.CATEGORICAL:
        if not defn.allowed_values:
            issues.append(ValidationIssue("categorical_no_values", "categorical parameters need allowed_values", "allowed_values"))

    for dc in defn.dependent_constraints:
        if not dc.driver_parameter.strip():
            issues.append(ValidationIssue("dependent_no_driver", "dependent constraint missing driver_parameter", "dependent_constraints"))

    return ParameterValidationResult(valid=len(issues) == 0, issues=issues)


def _driver_value_in_rule(
    driver_value: float,
    rule: DependentRangeRule,
) -> bool:
    if rule.driver_min is not None and driver_value < rule.driver_min - 1e-12:
        return False
    if rule.driver_max is not None and driver_value > rule.driver_max + 1e-12:
        return False
    return True


def _effective_bounds_for_value(
    defn: ParameterDefinition,
    driver_values: Optional[dict[str, float]],
) -> tuple[Optional[float], Optional[float]]:
    emin, emax = defn.effective_min(), defn.effective_max()
    if not defn.dependent_constraints or not driver_values:
        return emin, emax

    for dc in defn.dependent_constraints:
        dv = driver_values.get(dc.driver_parameter)
        if dv is None or math.isnan(dv):
            continue
        for rule in dc.rules:
            if _driver_value_in_rule(float(dv), rule):
                rmin = rule.min_value if rule.min_value is not None else emin
                rmax = rule.max_value if rule.max_value is not None else emax
                return rmin, rmax
    return emin, emax


def validate_parameter_value(
    defn: ParameterDefinition,
    value: Any,
    *,
    driver_values: Optional[dict[str, float]] = None,
) -> ParameterValidationResult:
    """Validate a single value against static bounds, step, and categorical/boolean rules."""
    issues: list[ValidationIssue] = []

    def_res = validate_parameter_definition(defn)
    if not def_res.valid:
        return def_res

    if not defn.enabled:
        return ParameterValidationResult(valid=True, issues=[])

    if defn.data_type == ParameterDataType.CATEGORICAL:
        allowed = defn.allowed_values
        if allowed is not None and value not in allowed:
            issues.append(ValidationIssue("not_in_allowed", f"value {value!r} not in allowed_values", "value"))
        return ParameterValidationResult(valid=len(issues) == 0, issues=issues)

    if defn.data_type == ParameterDataType.BOOLEAN:
        if value not in (True, False, 0, 1, 0.0, 1.0):
            issues.append(ValidationIssue("boolean_invalid", "boolean value must be true/false or 0/1", "value"))
        return ParameterValidationResult(valid=len(issues) == 0, issues=issues)

    if value is None or (isinstance(value, float) and math.isnan(value)):
        issues.append(ValidationIssue("value_missing", "numeric value is missing or NaN", "value"))
        return ParameterValidationResult(valid=False, issues=issues)

    try:
        v = float(value)
    except (TypeError, ValueError):
        issues.append(ValidationIssue("not_numeric", "value is not numeric", "value"))
        return ParameterValidationResult(valid=False, issues=issues)

    emin, emax = _effective_bounds_for_value(defn, driver_values)

    # Match constraints.py: skip hard min/max when bounds look like unset placeholders.
    bounds_set = emin is not None and emax is not None and not (float(emin) == 0.0 and float(emax) == 0.0)
    if bounds_set:
        if emin is not None and v < float(emin) - 1e-12:
            issues.append(ValidationIssue("below_min", f"value {v} below min {emin}", "value"))
        if emax is not None and v > float(emax) + 1e-12:
            issues.append(ValidationIssue("above_max", f"value {v} above max {emax}", "value"))

    step = defn.effective_step()
    if step > 0:
        if abs((v / step) - round(v / step)) > 1e-6:
            issues.append(ValidationIssue("not_on_step", f"value {v} is not aligned to step {step}", "value"))

    return ParameterValidationResult(valid=len(issues) == 0, issues=issues)


def validate_registry(reg: ParameterRegistry) -> ParameterValidationResult:
    all_issues: list[ValidationIssue] = []
    for d in reg._by_canonical.values():
        r = validate_parameter_definition(d)
        all_issues.extend(r.issues)
    return ParameterValidationResult(valid=len(all_issues) == 0, issues=all_issues)


def apply_parameter_registry_to_config(project_root: Path, config: dict[str, Any]) -> ParameterRegistry:
    """
    Merge structured registry files with legacy ``parameter_constraints`` from *config*.

    - Loads default + user registry JSON.
    - If registry files define parameters, uses their legacy projection as base.
    - Otherwise migrates from ``config['parameter_constraints']`` and column mapping.
    - Deep-merges legacy user overrides from ``config['parameter_constraints']`` on top.
    - Sets ``config['parameter_constraints']`` to the result.
    - Attaches ``config['_parameter_registry']`` for callers that need lookup/validation.
      The registry object reflects the same merged constraints as ``parameter_constraints``.
    """
    reg_disk = load_effective_parameter_registry(project_root)
    column_mapping = config.get("column_mapping", {}) or {}
    legacy_merged = config.get("parameter_constraints", {}) or {}

    if reg_disk._by_canonical:
        base_reg = reg_disk
        base_legacy = base_reg.to_legacy_parameter_constraints()
    else:
        base_reg = migrate_legacy_parameter_constraints(legacy_merged, column_mapping)
        base_legacy = base_reg.to_legacy_parameter_constraints()

    merged_pc = deep_merge(base_legacy, legacy_merged)
    config["parameter_constraints"] = merged_pc

    effective_reg = build_effective_registry(base_reg, merged_pc, column_mapping)
    config["_parameter_registry"] = effective_reg
    return effective_reg
