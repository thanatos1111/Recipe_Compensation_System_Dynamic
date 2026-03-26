"""
Instance correction layer (placeholder for Milestone 4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.feature_engineering import build_feature_matrix
from core.schemas import InstanceCorrectionArtifacts
from core.response_models import predict_outputs


def fit_instance_bias(
    material_model_bundle: dict[str, Any],
    instance_df: pd.DataFrame,
) -> Any:
    """Estimate instance-specific residual bias terms."""
    if instance_df is None or instance_df.empty:
        return InstanceCorrectionArtifacts(instance_confidence="Low")

    feature_config = material_model_bundle.get("feature_config", {})
    X = build_feature_matrix(instance_df, feature_config)
    preds = predict_outputs(material_model_bundle, X)

    bias_terms: dict[str, float] = {}
    recent_residual_summary: dict[str, Any] = {}

    for target_col, pred_key in [("rs", "rs_pred"), ("thickness", "thickness_pred"), ("rsu", "rsu_pred")]:
        if pred_key not in preds or target_col not in instance_df.columns:
            continue
        y = pd.to_numeric(instance_df[target_col], errors="coerce")
        y_pred = pd.Series(np.asarray(preds[pred_key]), index=instance_df.index)
        resid = (y - y_pred).dropna()
        if resid.empty:
            continue
        bias_terms[f"{target_col}_bias"] = float(resid.mean())
        recent_residual_summary[f"{target_col}_resid_mean"] = float(resid.mean())
        recent_residual_summary[f"{target_col}_resid_std"] = float(resid.std(ddof=0)) if len(resid) > 1 else 0.0

    conf = "Low"
    if len(instance_df) >= 5:
        conf = "Medium"
    if len(instance_df) >= 15:
        conf = "High"

    return InstanceCorrectionArtifacts(
        bias_terms=bias_terms,
        recent_residual_summary=recent_residual_summary,
        instance_confidence=conf,
        drift_summary={},
    )


def apply_instance_correction(predictions: dict[str, Any], correction_artifacts: Any) -> dict[str, Any]:
    """Apply correction artifacts to model predictions."""
    if predictions is None:
        return {}
    if correction_artifacts is None:
        return predictions

    out = dict(predictions)
    bias_terms = getattr(correction_artifacts, "bias_terms", None) or {}

    if "rs_pred" in out and "rs_bias" in bias_terms:
        out["rs_pred"] = np.asarray(out["rs_pred"], dtype=float) + float(bias_terms["rs_bias"])
    if "thickness_pred" in out and "thickness_bias" in bias_terms:
        out["thickness_pred"] = np.asarray(out["thickness_pred"], dtype=float) + float(bias_terms["thickness_bias"])
    if "rsu_pred" in out and "rsu_bias" in bias_terms:
        out["rsu_pred"] = np.asarray(out["rsu_pred"], dtype=float) + float(bias_terms["rsu_bias"])

    return out


def summarize_recent_residuals(instance_df: pd.DataFrame, material_model_bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize residual drift for the active target instance."""
    if instance_df is None or instance_df.empty:
        return {}

    n_recent = 5
    if "lifetime" in instance_df.columns:
        df_recent = instance_df.sort_values("lifetime").tail(n_recent)
    else:
        df_recent = instance_df.tail(n_recent)

    feature_config = material_model_bundle.get("feature_config", {})
    X = build_feature_matrix(df_recent, feature_config)
    preds = predict_outputs(material_model_bundle, X)

    out: dict[str, Any] = {"n_recent": int(len(df_recent))}
    for target_col, pred_key in [("rs", "rs_pred"), ("thickness", "thickness_pred"), ("rsu", "rsu_pred")]:
        if pred_key not in preds or target_col not in df_recent.columns:
            continue
        y = pd.to_numeric(df_recent[target_col], errors="coerce")
        y_pred = pd.Series(np.asarray(preds[pred_key]), index=df_recent.index)
        resid = (y - y_pred).dropna()
        if resid.empty:
            continue
        out[f"{target_col}_recent_resid_mean"] = float(resid.mean())

    return out

