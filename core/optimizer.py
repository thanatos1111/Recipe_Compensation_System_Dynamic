"""
Optimizer + scoring (placeholder for Milestone 5).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import pandas as pd

from core.constraints import validate_candidate_parameters
from core.feature_engineering import build_feature_matrix
from core.instance_correction import apply_instance_correction
from core.response_models import predict_outputs


def score_candidate(
    candidate: dict[str, Any],
    predicted_outputs: dict[str, Any],
    spec_config: Any,
    weights: dict[str, float],
    reference_recipe: dict[str, Any],
) -> dict[str, Any]:
    """Compute score breakdown for one candidate."""
    def _to_scalar(x: Any) -> Optional[float]:
        if x is None:
            return None
        # sklearn-like: predict returns array-like
        if hasattr(x, "shape") and getattr(x, "shape", None) is not None:
            try:
                if len(x) == 1:
                    return float(x[0])
            except Exception:
                pass
        # list/tuple
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                return float(x[0])
        # already scalar
        try:
            return float(x)
        except Exception:
            return None

    rs_pred = _to_scalar(predicted_outputs.get("rs_pred", None))
    thickness_pred = _to_scalar(predicted_outputs.get("thickness_pred", None))
    rsu_pred = _to_scalar(predicted_outputs.get("rsu_pred", None))

    w_rs = float(weights.get("w_rs", 1.0))
    w_thickness = float(weights.get("w_thickness", 1.0))
    w_rsu = float(weights.get("w_rsu", 1.0))
    w_move = float(weights.get("w_move", 0.1))
    w_margin = float(weights.get("w_margin", 0.1))

    # Move penalty: normalized by reference magnitude and step-free fallback.
    move_pen = 0.0
    for k, v in candidate.items():
        if k not in reference_recipe:
            continue
        if k in {"target_id"}:
            continue
        try:
            v0 = float(reference_recipe[k])
            dv = float(v) - v0
        except Exception:
            continue
        denom = max(abs(v0), 1e-9)
        move_pen += abs(dv) / denom

    # RS normalized error and safe margin
    rs_error_norm = 0.0
    rs_safe = 0.0
    if hasattr(spec_config, "use_rs_spec") and getattr(spec_config, "use_rs_spec", True) and rs_pred is not None:
        use_target_mode = bool(getattr(spec_config, "use_rs_target_mode", False))
        if use_target_mode:
            rs_target = getattr(spec_config, "rs_target", None)
            rs_tol = getattr(spec_config, "rs_tol", None)
            if rs_target is not None and rs_tol is not None and rs_tol != 0:
                rs_error_norm = abs(float(rs_pred) - float(rs_target)) / float(rs_tol)
                rs_safe = max(0.0, float(rs_tol) - abs(float(rs_pred) - float(rs_target)))
            else:
                rs_error_norm = 0.0
        else:
            rs_min = getattr(spec_config, "rs_min", None)
            rs_max = getattr(spec_config, "rs_max", None)
            if rs_min is not None and rs_max is not None and float(rs_max) != float(rs_min):
                if float(rs_pred) < float(rs_min):
                    rs_error_norm = (float(rs_min) - float(rs_pred)) / (abs(float(rs_max) - float(rs_min)))
                    rs_safe = 0.0
                elif float(rs_pred) > float(rs_max):
                    rs_error_norm = (float(rs_pred) - float(rs_max)) / (abs(float(rs_max) - float(rs_min)))
                    rs_safe = 0.0
                else:
                    # within range: safe reward proportional to distance to nearest boundary
                    rs_safe = min(float(rs_pred) - float(rs_min), float(rs_max) - float(rs_pred))
                    rs_error_norm = 0.0

    # Thickness error
    thickness_error_norm = 0.0
    thickness_safe = 0.0
    if hasattr(spec_config, "use_thickness_spec") and getattr(spec_config, "use_thickness_spec", True) and thickness_pred is not None:
        use_target_mode = bool(getattr(spec_config, "use_thickness_target_mode", False))
        if use_target_mode:
            thickness_target = getattr(spec_config, "thickness_target", None)
            thickness_tol = getattr(spec_config, "thickness_tol", None)
            if thickness_target is not None and thickness_tol is not None and thickness_tol != 0:
                thickness_error_norm = abs(float(thickness_pred) - float(thickness_target)) / float(thickness_tol)
                thickness_safe = max(0.0, float(thickness_tol) - abs(float(thickness_pred) - float(thickness_target)))
        else:
            tmin = getattr(spec_config, "thickness_min", None)
            tmax = getattr(spec_config, "thickness_max", None)
            if tmin is not None and tmax is not None and float(tmax) != float(tmin):
                if float(thickness_pred) < float(tmin):
                    thickness_error_norm = (float(tmin) - float(thickness_pred)) / (abs(float(tmax) - float(tmin)))
                elif float(thickness_pred) > float(tmax):
                    thickness_error_norm = (float(thickness_pred) - float(tmax)) / (abs(float(tmax) - float(tmin)))
                else:
                    thickness_safe = min(float(thickness_pred) - float(tmin), float(tmax) - float(thickness_pred))
                    thickness_error_norm = 0.0

    # RSU penalty
    rsu_pen = 0.0
    rsu_safe = 0.0
    if hasattr(spec_config, "use_rsu_spec") and getattr(spec_config, "use_rsu_spec", True) and rsu_pred is not None:
        rsu_max = float(getattr(spec_config, "rsu_max", 0.0) or 0.0)
        if rsu_max > 0:
            rsu_pen = max(0.0, float(rsu_pred) - rsu_max) / rsu_max
            rsu_safe = max(0.0, rsu_max - float(rsu_pred))

    safe_margin_reward = rsu_safe + rs_safe + thickness_safe
    # Normalize safe reward to avoid domination.
    safe_margin_scale = 1.0
    try:
        safe_margin_scale = max(abs(float(getattr(spec_config, "rsu_max", 1.0) or 1.0)), 1e-9)
    except Exception:
        safe_margin_scale = 1.0
    safe_margin_reward = safe_margin_reward / safe_margin_scale

    total = (
        w_rs * rs_error_norm
        + w_thickness * thickness_error_norm
        + w_rsu * rsu_pen
        + w_move * move_pen
        - w_margin * safe_margin_reward
    )

    return {
        "score": float(total),
        "rs_error_norm": float(rs_error_norm),
        "thickness_error_norm": float(thickness_error_norm),
        "rsu_penalty": float(rsu_pen),
        "move_penalty": float(move_pen),
        "safe_margin_reward": float(safe_margin_reward),
    }


def rank_candidates(
    candidates: pd.DataFrame,
    model_bundle: dict[str, Any],
    context: dict[str, Any],
) -> pd.DataFrame:
    """Predict, score, and rank candidates."""
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    spec_config = context.get("spec_config")
    weights = context.get("weights", {})
    reference_recipe = context.get("reference_recipe", {})

    feature_config = context.get("feature_config", {})
    active_target_id = context.get("active_target_id", None)
    lifetime = context.get("lifetime", None)

    # Machine context used for lifetime_used/end.
    rpm = context.get("rpm", None)
    power = context.get("power", None)

    correction = context.get("instance_correction", None)

    ranked_rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        cand = dict(row.to_dict())

        # Filter invalid/constraint-violating candidates.
        param_cfg = context.get("parameter_config", {})
        if param_cfg:
            violations = validate_candidate_parameters(cand, param_cfg)
            if violations:
                continue

        # Build feature row for the current candidate at requested lifetime.
        feature_row = {
            "target_id": active_target_id,
            "lifetime": lifetime,
            "incident_angle": cand.get("incident_angle", reference_recipe.get("incident_angle", None)),
            "linear_offset": cand.get("linear_offset", reference_recipe.get("linear_offset", None)),
            "rotations": cand.get("rotations", reference_recipe.get("rotations", None)),
            "ar_flow": cand.get("ar_flow", reference_recipe.get("ar_flow", None)),
            "o2_flow": cand.get("o2_flow", reference_recipe.get("o2_flow", None)),
            "rpm": rpm,
            "power": power,
        }

        # Derived features needed by feature_matrix.
        try:
            ar_flow = float(feature_row["ar_flow"]) if feature_row["ar_flow"] is not None else None
            o2_flow = float(feature_row["o2_flow"]) if feature_row["o2_flow"] is not None else None
            if ar_flow is not None and o2_flow is not None:
                total_flow = ar_flow + o2_flow
                feature_row["total_flow"] = total_flow
                feature_row["o2_ratio"] = (o2_flow / total_flow) if total_flow != 0 else pd.NA
            else:
                feature_row["total_flow"] = pd.NA
                feature_row["o2_ratio"] = pd.NA
        except Exception:
            feature_row["total_flow"] = pd.NA
            feature_row["o2_ratio"] = pd.NA

        try:
            rotations = float(feature_row["rotations"]) if feature_row["rotations"] is not None else None
            rpm_f = float(feature_row["rpm"]) if feature_row["rpm"] is not None else None
            power_f = float(feature_row["power"]) if feature_row["power"] is not None else None
            lifetime_start = float(feature_row["lifetime"]) if feature_row["lifetime"] is not None else None
            if rotations is not None and rpm_f is not None and rpm_f > 0 and power_f is not None and lifetime_start is not None:
                duration_hours = rotations / rpm_f / 60.0
                lifetime_used = power_f / 1000.0 * duration_hours
                feature_row["lifetime_used"] = lifetime_used
                feature_row["lifetime_end"] = lifetime_start + lifetime_used
            else:
                feature_row["lifetime_used"] = pd.NA
                feature_row["lifetime_end"] = pd.NA
        except Exception:
            feature_row["lifetime_used"] = pd.NA
            feature_row["lifetime_end"] = pd.NA

        X = pd.DataFrame([feature_row])
        X_feat = build_feature_matrix(X, feature_config)

        predicted = predict_outputs(model_bundle, X_feat)
        predicted_corr = apply_instance_correction(predicted, correction)

        score_breakdown = score_candidate(
            cand,
            predicted_corr,
            spec_config,
            weights=weights,
            reference_recipe=reference_recipe,
        )

        out_row = dict(cand)
        out_row.update(predicted_corr)
        out_row.update(score_breakdown)
        ranked_rows.append(out_row)

    ranked_df = pd.DataFrame(ranked_rows)
    if not ranked_df.empty and "score" in ranked_df.columns:
        ranked_df = ranked_df.sort_values("score").reset_index(drop=True)
    return ranked_df


def select_best_candidate(ranked_df: pd.DataFrame) -> dict[str, Any]:
    """Select the best candidate (top row) from ranked candidates."""
    if ranked_df is None or ranked_df.empty:
        return {}
    top = ranked_df.iloc[0].to_dict()
    return top

