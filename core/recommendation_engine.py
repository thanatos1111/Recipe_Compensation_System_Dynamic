"""
Programmatic recommendation engine used by both the G panel and the update panel.

Milestone 6 helper: keep UI duplication low and provide consistent outputs.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.candidate_generator import generate_candidates
from core.optimizer import rank_candidates, select_best_candidate
from core.schemas import SpecConfig


def _effective_parameter_config(config: dict[str, Any]) -> dict[str, Any]:
    parameter_config = config.get("parameter_constraints", {}) or {}
    min_steps = config.get("minimum_steps", {}) or {}

    def default_min_step(param_name: str) -> float:
        return 1.0 if param_name == "rotations" else 0.01

    effective: dict[str, Any] = {}
    for pname, pcfg in (parameter_config or {}).items():
        if not isinstance(pcfg, dict):
            effective[pname] = pcfg
            continue
        eff = dict(pcfg)
        current_step = float(eff.get("step", 0.0) or 0.0)
        ms = float(min_steps.get(pname, default_min_step(pname)) or 0.0)
        if current_step <= 0:
            eff["step"] = ms
        else:
            eff["step"] = max(current_step, ms)
        effective[pname] = eff
    return effective


def recommend_next_recipe(
    *,
    material_dataset: Any,
    active_target_id: str,
    lifetime_start: float,
    config: dict[str, Any],
    neighborhood_config: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns (ranked_df, best_candidate_dict).

    Notes:
    - `lifetime_start` is the requested starting lifetime for the deposition state.
    - Candidate search is conservative local search around the latest actual recipe.
    """
    if material_dataset is None or active_target_id not in material_dataset.target_instances:
        return pd.DataFrame(), {}

    artifacts = getattr(material_dataset, "material_model_artifacts", None)
    if artifacts is None or artifacts.rs_model is None or artifacts.thickness_model is None or artifacts.rsu_model is None:
        return pd.DataFrame(), {}

    inst = material_dataset.target_instances[active_target_id]
    if inst.records is None or inst.records.empty:
        return pd.DataFrame(), {}

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

    spec_config: SpecConfig = getattr(material_dataset, "spec_config", None)
    parameter_config = _effective_parameter_config(config)
    neighborhood_config = neighborhood_config or {"radius_steps": 1, "max_candidates": 200}

    candidates = generate_candidates(reference_recipe, parameter_config, neighborhood_config)
    if candidates is None or candidates.empty:
        return pd.DataFrame(), {}

    model_bundle = {
        "rs_model": artifacts.rs_model,
        "thickness_model": artifacts.thickness_model,
        "rsu_model": artifacts.rsu_model,
        "feature_config": config.get("feature_config", {}),
        "instance_correction": inst.instance_correction_artifacts,
    }

    context = {
        "spec_config": spec_config,
        "weights": config.get("optimizer_weights", {}),
        "reference_recipe": reference_recipe,
        "active_target_id": active_target_id,
        "lifetime": float(lifetime_start),
        "rpm": rpm,
        "power": power,
        "feature_config": config.get("feature_config", {}),
        "instance_correction": inst.instance_correction_artifacts,
        "parameter_config": parameter_config,
    }

    ranked_df = rank_candidates(candidates, model_bundle, context)
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame(), {}

    best = select_best_candidate(ranked_df)
    return ranked_df, best

