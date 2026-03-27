"""
Candidate recipe generation (placeholder for Milestone 5).
"""

from __future__ import annotations

from itertools import product
from copy import deepcopy
from typing import Any

import pandas as pd

from core.constraints import enforce_coupling_rules, enforce_discrete_step


def generate_candidates(
    reference_recipe: dict[str, Any],
    parameter_config: dict[str, Any],
    neighborhood_config: dict[str, Any],
) -> pd.DataFrame:
    """Generate discrete candidate recipes around a reference point."""
    radius_steps = int(neighborhood_config.get("radius_steps", 1))
    max_candidates = int(neighborhood_config.get("max_candidates", 200))

    min_steps: dict[str, Any] = neighborhood_config.get("min_steps", {}) or {}

    def get_min_step(param_name: str) -> float:
        if param_name in min_steps:
            try:
                return float(min_steps[param_name])
            except Exception:
                pass
        # Defaults from design request.
        return 1.0 if param_name == "rotations" else 0.01

    effective_parameter_config = deepcopy(parameter_config or {})
    for p, cfg in (effective_parameter_config or {}).items():
        if not isinstance(cfg, dict):
            continue
        if not cfg.get("is_enabled", True):
            continue
        current_step = float(cfg.get("step", 0.0) or 0.0)
        ms = float(get_min_step(p) or 0.0)
        if current_step <= 0:
            cfg["step"] = ms
        else:
            cfg["step"] = max(current_step, ms)

    enabled_params = [
        k
        for k, v in (effective_parameter_config or {}).items()
        if isinstance(v, dict) and v.get("is_enabled", True)
    ]

    # Coupling handling: if both ar_flow and o2_flow are coupled by total_flow, vary o2_flow and derive ar_flow.
    coupling_group = None
    coupled_group_params: list[str] = []
    for p in enabled_params:
        cfg = effective_parameter_config.get(p, {})
        if cfg.get("is_coupled") and cfg.get("coupling_group"):
            coupling_group = str(cfg.get("coupling_group"))
            coupled_group_params.append(p)
    coupled_group_params = sorted(set(coupled_group_params))

    # Determine total_flow reference when applicable.
    total_flow_value = None
    if coupling_group == "total_flow":
        ar = reference_recipe.get("ar_flow", None)
        o2 = reference_recipe.get("o2_flow", None)
        if ar is not None and o2 is not None:
            total_flow_value = float(ar) + float(o2)

    varying_param_values: dict[str, list[Any]] = {}

    # Build candidate lists for each independent parameter.
    for p in enabled_params:
        # Skip coupled parameters if we'll derive them.
        if coupling_group == "total_flow" and p in {"ar_flow"}:
            continue
        cfg = effective_parameter_config.get(p, {})
        ref_val = reference_recipe.get(p, None)
        if ref_val is None:
            continue

        step = float(cfg.get("step", 0.0) or 0.0)
        if step <= 0:
            step = get_min_step(p)

        vals: list[Any] = []
        for k in range(-radius_steps, radius_steps + 1):
            v = float(ref_val) + k * step
            vals.append(v)

        # De-dup while preserving order.
        dedup: list[Any] = []
        seen = set()
        for v in vals:
            vv = float(v)
            if vv not in seen:
                seen.add(vv)
                dedup.append(vv)
        varying_param_values[p] = dedup

    # If no candidates vary, return the reference only.
    if not varying_param_values:
        return pd.DataFrame([reference_recipe])

    # Cartesian product across independent varied parameters.
    keys = list(varying_param_values.keys())
    value_lists = [varying_param_values[k] for k in keys]

    rows: list[dict[str, Any]] = []
    for combo in product(*value_lists):
        cand = dict(reference_recipe)
        for k, v in zip(keys, combo):
            cand[k] = v

        # Quantize/discretize.
        cand = enforce_discrete_step(cand, effective_parameter_config)
        # Enforce coupling for coupled groups.
        if coupling_group == "total_flow" and total_flow_value is not None:
            cand = enforce_coupling_rules(
                cand,
                {"coupling_group": "total_flow", "total_flow_value": total_flow_value},
            )

        rows.append(cand)
        if len(rows) >= max_candidates:
            break

    return pd.DataFrame(rows)


def apply_coupling_rules(candidate: dict[str, Any], coupling_config: dict[str, Any]) -> dict[str, Any]:
    """Enforce coupling rules on a candidate recipe."""
    return enforce_coupling_rules(candidate, coupling_config)


def is_valid_candidate(candidate: dict[str, Any], parameter_config: dict[str, Any]) -> bool:
    """Return whether the candidate satisfies discrete constraints."""
    from core.constraints import validate_candidate_parameters

    violations = validate_candidate_parameters(candidate, parameter_config)
    return len(violations) == 0

