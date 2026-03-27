"""
Feasible-region / safe-band estimation (placeholder).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.feature_engineering import build_feature_matrix
from core.instance_correction import apply_instance_correction
from core.response_models import predict_outputs


def _predicted_within_spec(predicted_outputs: dict[str, Any], spec_config: Any) -> bool:
    def _to_scalar(x: Any):
        if x is None:
            return None
        if hasattr(x, "shape") and getattr(x, "shape", None) is not None:
            try:
                if len(x) == 1:
                    return float(x[0])
            except Exception:
                pass
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return float(x[0])
        try:
            return float(x)
        except Exception:
            return None

    rsu_pred = _to_scalar(predicted_outputs.get("rsu_pred", None))
    rs_pred = _to_scalar(predicted_outputs.get("rs_pred", None))
    thickness_pred = _to_scalar(predicted_outputs.get("thickness_pred", None))

    # RSU
    if getattr(spec_config, "use_rsu_spec", True) and rsu_pred is not None:
        rsu_max = float(getattr(spec_config, "rsu_max", 0.0) or 0.0)
        if rsu_max > 0 and float(rsu_pred) > rsu_max:
            return False

    # RS
    if getattr(spec_config, "use_rs_spec", True) and rs_pred is not None:
        if getattr(spec_config, "use_rs_target_mode", False):
            rs_target = getattr(spec_config, "rs_target", None)
            rs_tol = getattr(spec_config, "rs_tol", None)
            if rs_target is not None and rs_tol is not None and rs_tol > 0:
                if abs(float(rs_pred) - float(rs_target)) > float(rs_tol):
                    return False
        else:
            rs_min = getattr(spec_config, "rs_min", None)
            rs_max = getattr(spec_config, "rs_max", None)
            if rs_min is not None and float(rs_pred) < float(rs_min):
                return False
            if rs_max is not None and float(rs_pred) > float(rs_max):
                return False

    # Thickness
    if getattr(spec_config, "use_thickness_spec", True) and thickness_pred is not None:
        if getattr(spec_config, "use_thickness_target_mode", False):
            t_target = getattr(spec_config, "thickness_target", None)
            t_tol = getattr(spec_config, "thickness_tol", None)
            if t_target is not None and t_tol is not None and t_tol > 0:
                if abs(float(thickness_pred) - float(t_target)) > float(t_tol):
                    return False
        else:
            tmin = getattr(spec_config, "thickness_min", None)
            tmax = getattr(spec_config, "thickness_max", None)
            if tmin is not None and float(thickness_pred) < float(tmin):
                return False
            if tmax is not None and float(thickness_pred) > float(tmax):
                return False

    return True


def estimate_parameter_band(
    center_recipe: dict[str, Any],
    model_bundle: dict[str, Any],
    *,
    parameter_name: str,
    spec_config: Any,
    step: float,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Estimate acceptable parameter interval around a center recipe."""
    if step is None or float(step) <= 0:
        return {"parameter_name": parameter_name, "lower": None, "upper": None, "passed_values": []}

    base = dict(center_recipe)
    passed: list[float] = []

    for k in range(-max_steps, max_steps + 1):
        cand_val = float(base.get(parameter_name, 0.0)) + k * float(step)
        cand = dict(base)
        cand[parameter_name] = cand_val

        lifetime = base.get("lifetime", None)
        rpm = base.get("rpm", None)
        power = base.get("power", None)
        target_id = base.get("target_id", None)

        feature_row = {
            "target_id": target_id,
            "lifetime": lifetime,
            "incident_angle": cand.get("incident_angle", base.get("incident_angle", None)),
            "linear_offset": cand.get("linear_offset", base.get("linear_offset", None)),
            "rotations": cand.get("rotations", base.get("rotations", None)),
            "ar_flow": cand.get("ar_flow", base.get("ar_flow", None)),
            "o2_flow": cand.get("o2_flow", base.get("o2_flow", None)),
            "rpm": rpm,
            "power": power,
        }

        # Derived fields needed by feature schema.
        try:
            ar_flow = float(feature_row["ar_flow"])
            o2_flow = float(feature_row["o2_flow"])
            total_flow = ar_flow + o2_flow
            feature_row["total_flow"] = total_flow
            feature_row["o2_ratio"] = (o2_flow / total_flow) if total_flow != 0 else pd.NA
        except Exception:
            feature_row["total_flow"] = pd.NA
            feature_row["o2_ratio"] = pd.NA

        try:
            rotations = float(feature_row["rotations"])
            rpm_f = float(rpm) if rpm is not None else None
            power_f = float(power) if power is not None else None
            lifetime_start = float(lifetime) if lifetime is not None else None
            if rpm_f and rpm_f > 0 and power_f is not None and lifetime_start is not None:
                duration_hours = rotations / rpm_f / 60.0
                feature_row["lifetime_used"] = float(power_f) / 1000.0 * duration_hours
                feature_row["lifetime_end"] = lifetime_start + float(feature_row["lifetime_used"])
            else:
                feature_row["lifetime_used"] = pd.NA
                feature_row["lifetime_end"] = pd.NA
        except Exception:
            feature_row["lifetime_used"] = pd.NA
            feature_row["lifetime_end"] = pd.NA

        X = pd.DataFrame([feature_row])
        feature_config = model_bundle.get("feature_config", {})
        X_feat = build_feature_matrix(X, feature_config)

        predicted = predict_outputs(model_bundle, X_feat)
        correction = model_bundle.get("instance_correction", None)
        if correction is not None:
            predicted = apply_instance_correction(predicted, correction)

        if _predicted_within_spec(predicted, spec_config):
            passed.append(float(cand_val))

    if not passed:
        return {"parameter_name": parameter_name, "lower": None, "upper": None, "passed_values": []}

    return {
        "parameter_name": parameter_name,
        "lower": min(passed),
        "upper": max(passed),
        "passed_values": passed,
    }


def build_feasible_map(
    center_recipe: dict[str, Any],
    model_bundle: dict[str, Any],
    varying_parameters: list[str],
    *,
    spec_config: Any,
) -> dict[str, Any]:
    """Build a feasible map (placeholder)."""
    out: dict[str, Any] = {}
    for p in varying_parameters:
        step = float(center_recipe.get(f"{p}_step", 0.0) or 0.0)
        if step <= 0:
            step = 1.0
        out[p] = estimate_parameter_band(
            center_recipe,
            model_bundle,
            parameter_name=p,
            spec_config=spec_config,
            step=step,
            max_steps=int(center_recipe.get("band_max_steps", 3)),
        )
    return out

