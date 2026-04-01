"""
Constraint checks for discrete recipe optimization (placeholder).
"""

from __future__ import annotations

import math
from typing import Any


def _value_in_driver_span(driver_value: float, rule: dict[str, Any]) -> bool:
    dmin = rule.get("driver_min", None)
    dmax = rule.get("driver_max", None)
    if dmin is not None and driver_value < float(dmin) - 1e-12:
        return False
    if dmax is not None and driver_value > float(dmax) + 1e-12:
        return False
    return True


def _effective_bounds_from_dependent_rules(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[Any, Any]:
    """
    Resolve effective (min_value, max_value) for a parameter config dict, honoring
    a legacy ``dependent_constraints`` list when present.
    """
    min_value = cfg.get("min_value", None)
    max_value = cfg.get("max_value", None)
    dcs = cfg.get("dependent_constraints")
    if not isinstance(dcs, list) or not dcs:
        return min_value, max_value
    for dc in dcs:
        if not isinstance(dc, dict):
            continue
        driver = str(dc.get("driver_parameter", "")).strip()
        if not driver:
            continue
        if driver not in candidate:
            continue
        dv_raw = candidate.get(driver)
        if dv_raw is None or (isinstance(dv_raw, float) and math.isnan(dv_raw)):
            continue
        try:
            dv = float(dv_raw)
        except (TypeError, ValueError):
            continue
        rules = dc.get("rules")
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if _value_in_driver_span(dv, rule):
                rmin = rule.get("min_value", None)
                rmax = rule.get("max_value", None)
                return (min_value if rmin is None else rmin, max_value if rmax is None else rmax)
    return min_value, max_value


def validate_candidate_parameters(candidate: dict[str, Any], parameter_config: dict[str, Any]) -> list[str]:
    """Return a list of constraint violation messages."""
    violations: list[str] = []

    for param_name, cfg in (parameter_config or {}).items():
        if not isinstance(cfg, dict):
            continue
        if not cfg.get("is_enabled", True):
            continue
        if param_name not in candidate:
            continue

        value = candidate.get(param_name)

        allowed_values = cfg.get("allowed_values", None)
        if allowed_values is not None:
            if value not in allowed_values:
                violations.append(f"{param_name}_not_allowed")
                continue

        step = float(cfg.get("step", 0.0) or 0.0)
        min_value, max_value = _effective_bounds_from_dependent_rules(cfg, candidate)

        # If bounds look like placeholders (0/0), don't hard-enforce.
        bounds_set = min_value is not None and max_value is not None and not (
            float(min_value) == 0.0 and float(max_value) == 0.0
        )
        if bounds_set and value is not None and not (isinstance(value, float) and math.isnan(value)):
            v = float(value)
            if min_value is not None and v < float(min_value) - 1e-12:
                violations.append(f"{param_name}_below_min")
            if max_value is not None and v > float(max_value) + 1e-12:
                violations.append(f"{param_name}_above_max")

        # Discrete step check (tolerant).
        if step > 0 and value is not None and not (isinstance(value, float) and math.isnan(value)):
            v = float(value)
            # Allow tiny numerical drift.
            if abs((v / step) - round(v / step)) > 1e-6:
                violations.append(f"{param_name}_not_on_step")

    return violations


def enforce_discrete_step(candidate: dict[str, Any], parameter_config: dict[str, Any]) -> dict[str, Any]:
    """Quantize candidate parameters to valid discrete step sizes."""
    out = dict(candidate)

    for param_name, cfg in (parameter_config or {}).items():
        if param_name not in out:
            continue
        if not isinstance(cfg, dict):
            continue
        step = float(cfg.get("step", 0.0) or 0.0)
        min_value, max_value = _effective_bounds_from_dependent_rules(cfg, out)

        if step <= 0:
            continue
        if out[param_name] is None:
            continue

        v = float(out[param_name])
        q = round(v / step) * step

        # Clamp only when bounds appear meaningful.
        if min_value is not None and max_value is not None and not (
            float(min_value) == 0.0 and float(max_value) == 0.0
        ):
            q = min(q, float(max_value))
            q = max(q, float(min_value))

        # Preserve integer type where relevant.
        p_type = str(cfg.get("type", "continuous"))
        if p_type == "integer":
            q = int(round(q))

        out[param_name] = q

    return out


def enforce_coupling_rules(candidate: dict[str, Any], coupling_config: dict[str, Any]) -> dict[str, Any]:
    """Enforce coupling rules like total flow constancy."""
    out = dict(candidate)

    group = coupling_config.get("coupling_group", None)
    # For this milestone, only implement total_flow = ar_flow + o2_flow constant.
    if group == "total_flow":
        total = coupling_config.get("total_flow_value", None)
        if total is None:
            return out
        if "o2_flow" in out:
            out["ar_flow"] = float(total) - float(out["o2_flow"])
        elif "ar_flow" in out:
            out["o2_flow"] = float(total) - float(out["ar_flow"])

    return out

