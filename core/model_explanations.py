"""
Human-readable training explanation builders for the Model tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class ExplanationItem:
    key: str
    label: str
    value: str
    detail: str
    tooltip: str


@dataclass(frozen=True)
class TrainingExplanation:
    summary: str
    items: list[ExplanationItem]
    warnings: list[str]


def build_training_explanation(
    *,
    train_df: Optional[pd.DataFrame],
    artifacts: Any,
    active_target_id: Optional[str],
    feature_schema: dict[str, list[str]],
) -> TrainingExplanation:
    """
    Build dynamic training explanations from model state and data context.
    """
    if train_df is None or train_df.empty:
        return TrainingExplanation(
            summary="No training rows are available, so model fitting and validation cannot run yet.",
            items=[
                ExplanationItem(
                    key="training_rows",
                    label="Training row count",
                    value="0",
                    detail="Add rows for this material and selected targets to enable training.",
                    tooltip="Number of rows that pass current training target selection.",
                )
            ],
            warnings=["No data to train on."],
        )

    row_count = int(len(train_df))
    in_spec_count: Optional[int] = None
    out_spec_count: Optional[int] = None
    if "in_spec" in train_df.columns:
        in_spec_series = train_df["in_spec"].astype(bool)
        in_spec_count = int(in_spec_series.sum())
        out_spec_count = int(row_count - in_spec_count)

    target_count = int(train_df["target_id"].astype(str).nunique()) if "target_id" in train_df.columns else 0
    sparse = row_count < 25
    very_sparse = row_count < 10

    metrics = (artifacts.metrics or {}) if artifacts is not None else {}
    train_summary = (artifacts.train_summary or {}) if artifacts is not None else {}
    confidence_summary = (artifacts.confidence_summary or {}) if artifacts is not None else {}

    rs_model = getattr(artifacts, "rs_model", None) if artifacts is not None else None
    thickness_model = getattr(artifacts, "thickness_model", None) if artifacts is not None else None
    rsu_model = getattr(artifacts, "rsu_model", None) if artifacts is not None else None

    fallback_responses: list[str] = []
    if rs_model is None:
        fallback_responses.append("RS")
    if thickness_model is None:
        fallback_responses.append("Thickness")
    if rsu_model is None:
        fallback_responses.append("RSU")

    historical_df = train_df
    if active_target_id and "target_id" in train_df.columns:
        historical_df = train_df[train_df["target_id"].astype(str) != str(active_target_id)]
        if historical_df.empty:
            historical_df = train_df
    training_lifetime = _lifetime_range(historical_df)
    active_lifetime = _active_target_lifetime_range(train_df, active_target_id)
    extrapolation_warning = _extrapolation_warning(training_lifetime, active_lifetime)

    feature_list = feature_schema.get("numeric", []) + feature_schema.get("categorical", [])
    feature_text = ", ".join(feature_list) if feature_list else "No engineered features available"

    model_type_text = _describe_model_types(rs_model, thickness_model, rsu_model)
    validation_text = _describe_validation(metrics)
    mae_rmse_text = _describe_error_metrics(metrics)
    feature_importance_text = _describe_feature_importance(rs_model, thickness_model, rsu_model)
    residual_text = _describe_residuals(rs_model, thickness_model, rsu_model)
    confidence_text = _describe_confidence(confidence_summary, sparse=sparse, very_sparse=very_sparse)

    items = [
        ExplanationItem(
            key="training_rows",
            label="Training row count",
            value=str(row_count),
            detail=f"Rows come from selected targets in one material; target instances represented: {target_count}.",
            tooltip="How many rows were used to fit the model.",
        ),
        ExplanationItem(
            key="in_spec_counts",
            label="In-spec / out-of-spec count",
            value=_fmt_in_out(in_spec_count, out_spec_count),
            detail="Both in-spec and out-of-spec rows are retained to learn full process behavior.",
            tooltip="In-spec means rows meeting current spec limits; out-of-spec means they do not.",
        ),
        ExplanationItem(
            key="feature_list",
            label="Feature list",
            value=feature_text,
            detail="These are the engineered inputs used as predictors for RS/Thickness/RSU.",
            tooltip="Numeric and categorical predictors after feature engineering.",
        ),
        ExplanationItem(
            key="model_type",
            label="Model type",
            value=model_type_text,
            detail=(
                "Fallback mode is active for: " + ", ".join(fallback_responses)
                if fallback_responses
                else "Primary configured models are fitted for all responses."
            ),
            tooltip="Model family used per response. Missing model indicates fallback/unavailable training.",
        ),
        ExplanationItem(
            key="validation_method",
            label="Validation method",
            value=validation_text,
            detail="Uses a time-aware split so later rows are used as holdout to mimic future prediction.",
            tooltip="How train/test rows are separated for reported metrics.",
        ),
        ExplanationItem(
            key="mae_rmse",
            label="MAE / RMSE",
            value=mae_rmse_text,
            detail="Lower values are better; MAE is average absolute error, RMSE penalizes larger errors more strongly.",
            tooltip="Core regression error metrics for each response.",
        ),
        ExplanationItem(
            key="feature_importance",
            label="Feature importance",
            value=feature_importance_text,
            detail="Importance explains which features influence predictions most for each response model.",
            tooltip="Derived from model importances or coefficient magnitudes when available.",
        ),
        ExplanationItem(
            key="residual_plots",
            label="Residual plots",
            value=residual_text,
            detail="Residual distributions centered around zero indicate lower bias and better fit balance.",
            tooltip="Residual = actual - predicted. Look for centered and tight distributions.",
        ),
        ExplanationItem(
            key="confidence_summary",
            label="Confidence summary",
            value=confidence_text,
            detail="Confidence combines data volume and coverage signals to qualify model trust level.",
            tooltip="High confidence means broader support from data and coverage checks.",
        ),
    ]

    warnings: list[str] = []
    if sparse:
        warnings.append("Sparse-data condition: limited rows may reduce stability.")
    if very_sparse:
        warnings.append("Very sparse data: one or more response models may be unavailable.")
    if fallback_responses:
        warnings.append("Fallback mode active for: " + ", ".join(fallback_responses) + ".")
    if extrapolation_warning:
        warnings.append(extrapolation_warning)

    conf_level = str(confidence_summary.get("level", "unknown")).lower()
    summary = (
        "The model learns relationships between process features and RS/Thickness/RSU using "
        f"{row_count} rows across {target_count} target instances for this material. "
        f"Validation is {validation_text}. "
        f"Current confidence is {conf_level}."
    )
    if warnings:
        summary = summary + " " + " ".join(warnings)

    # Keep explicit display of train summary fields when available.
    if train_summary:
        train_row = train_summary.get("row_count")
        if train_row is not None and int(train_row) != row_count:
            warnings.append(
                f"Displayed training rows ({row_count}) differ from artifact summary rows ({train_row})."
            )

    return TrainingExplanation(summary=summary, items=items, warnings=warnings)


def _fmt_in_out(in_spec_count: Optional[int], out_spec_count: Optional[int]) -> str:
    if in_spec_count is None or out_spec_count is None:
        return "in-spec / out-of-spec unavailable"
    return f"{in_spec_count} in-spec / {out_spec_count} out-of-spec"


def _lifetime_range(df: pd.DataFrame) -> Optional[tuple[float, float]]:
    if "lifetime" not in df.columns:
        return None
    lifetime = pd.to_numeric(df["lifetime"], errors="coerce").dropna()
    if lifetime.empty:
        return None
    return float(lifetime.min()), float(lifetime.max())


def _active_target_lifetime_range(df: pd.DataFrame, active_target_id: Optional[str]) -> Optional[tuple[float, float]]:
    if not active_target_id or "target_id" not in df.columns:
        return None
    sub = df[df["target_id"].astype(str) == str(active_target_id)]
    if sub.empty:
        return None
    return _lifetime_range(sub)


def _extrapolation_warning(
    training_lifetime: Optional[tuple[float, float]],
    active_lifetime: Optional[tuple[float, float]],
) -> str:
    if training_lifetime is None or active_lifetime is None:
        return ""
    t_min, t_max = training_lifetime
    a_min, a_max = active_lifetime
    if a_min < t_min or a_max > t_max:
        return (
            f"Extrapolation warning: active target lifetime range [{a_min:.4g}, {a_max:.4g}] "
            f"extends beyond training range [{t_min:.4g}, {t_max:.4g}]."
        )
    return ""


def _describe_model_types(rs_model: Any, thickness_model: Any, rsu_model: Any) -> str:
    return ", ".join(
        [
            f"RS={_model_name(rs_model)}",
            f"Thickness={_model_name(thickness_model)}",
            f"RSU={_model_name(rsu_model)}",
        ]
    )


def _model_name(model: Any) -> str:
    if model is None:
        return "unavailable"
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return type(model.named_steps["model"]).__name__
    return type(model).__name__


def _describe_validation(metrics: dict[str, Any]) -> str:
    n_train = metrics.get("n_train")
    n_test = metrics.get("n_test")
    frac = metrics.get("test_fraction")
    if n_train is None or n_test is None:
        return "time-aware holdout (pending)"
    if frac is None:
        return f"time-aware holdout (train={n_train}, test={n_test})"
    return f"time-aware holdout (train={n_train}, test={n_test}, test_fraction={float(frac):.2f})"


def _describe_error_metrics(metrics: dict[str, Any]) -> str:
    rs_mae = metrics.get("rs_mae")
    th_mae = metrics.get("thickness_mae")
    rsu_mae = metrics.get("rsu_mae")
    rs_rmse = metrics.get("rs_rmse")
    th_rmse = metrics.get("thickness_rmse")
    rsu_rmse = metrics.get("rsu_rmse")
    return (
        f"RS: MAE={_fmt_metric(rs_mae)}, RMSE={_fmt_metric(rs_rmse)} | "
        f"Thickness: MAE={_fmt_metric(th_mae)}, RMSE={_fmt_metric(th_rmse)} | "
        f"RSU: MAE={_fmt_metric(rsu_mae)}, RMSE={_fmt_metric(rsu_rmse)}"
    )


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return "n/a"


def _describe_feature_importance(rs_model: Any, thickness_model: Any, rsu_model: Any) -> str:
    return ", ".join(
        [
            f"RS={_importance_mode(rs_model)}",
            f"Thickness={_importance_mode(thickness_model)}",
            f"RSU={_importance_mode(rsu_model)}",
        ]
    )


def _importance_mode(model: Any) -> str:
    if model is None:
        return "n/a"
    est = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(est, "feature_importances_"):
        return "tree importance"
    if hasattr(est, "coef_"):
        return "coefficient magnitude"
    return "unsupported"


def _describe_residuals(rs_model: Any, thickness_model: Any, rsu_model: Any) -> str:
    ready = []
    for name, model in [("RS", rs_model), ("Thickness", thickness_model), ("RSU", rsu_model)]:
        if model is not None:
            ready.append(name)
    if not ready:
        return "Residual plots unavailable until at least one response model is fitted."
    return "Residual diagnostics available for: " + ", ".join(ready)


def _describe_confidence(confidence_summary: dict[str, Any], *, sparse: bool, very_sparse: bool) -> str:
    level = str(confidence_summary.get("level", "unknown"))
    notes = confidence_summary.get("notes", [])
    note_text = ", ".join(str(v) for v in notes) if isinstance(notes, list) and notes else "none"
    suffix = ""
    if very_sparse:
        suffix = " (very sparse)"
    elif sparse:
        suffix = " (sparse)"
    return f"level={level}{suffix}; notes={note_text}"

