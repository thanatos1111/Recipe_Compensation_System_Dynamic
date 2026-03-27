"""
Material-specific response model training + prediction (placeholder).

Milestone 4 will implement model training and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.feature_engineering import FeatureSchema, build_feature_matrix, get_feature_schema
from core.schemas import MaterialModelArtifacts


def train_rs_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RS regression model."""
    return _train_regression_model(df, feature_matrix, target_col="rs", config=config)


def train_thickness_model(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Train a thickness regression model."""
    return _train_regression_model(df, feature_matrix, target_col="thickness", config=config)


def train_rsu_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RSU regression model."""
    return _train_regression_model(df, feature_matrix, target_col="rsu", config=config)


def train_spec_classifier(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Optional classifier for in-spec probability."""
    # Not implemented in Milestone 4 by default (optional).
    del df, feature_matrix, config
    return None


def predict_outputs(model_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Predict RS, thickness, and RSU (and optional in-spec probability)."""
    rs_model = model_bundle.get("rs_model")
    thickness_model = model_bundle.get("thickness_model")
    rsu_model = model_bundle.get("rsu_model")

    out: dict[str, Any] = {}
    if rs_model is not None:
        out["rs_pred"] = rs_model.predict(X)
    if thickness_model is not None:
        out["thickness_pred"] = thickness_model.predict(X)
    if rsu_model is not None:
        out["rsu_pred"] = rsu_model.predict(X)
    return out


def train_material_models(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> MaterialModelArtifacts:
    """
    Train per-material models for RS, thickness, and RSU.

    Uses all rows (including out-of-spec) as required by the design spec.
    """
    feature_config = config.get("feature_config", {})
    schema = get_feature_schema(feature_config)
    X = build_feature_matrix(df, feature_config)

    rs_model = train_rs_model(df, X, config)
    thickness_model = train_thickness_model(df, X, config)
    rsu_model = train_rsu_model(df, X, config)

    metrics = evaluate_models_time_aware(
        df,
        X,
        rs_model=rs_model,
        thickness_model=thickness_model,
        rsu_model=rsu_model,
        config=config,
    )

    train_summary = {
        "row_count": int(len(df)),
        "in_spec_count": int(df["in_spec"].sum()) if "in_spec" in df.columns else None,
        "target_instance_count": int(df["target_id"].nunique()) if "target_id" in df.columns else None,
    }

    confidence_summary = {
        "level": _confidence_level(df),
        "notes": _confidence_notes(df),
    }

    return MaterialModelArtifacts(
        rs_model=rs_model,
        thickness_model=thickness_model,
        rsu_model=rsu_model,
        spec_classifier=None,
        feature_schema={"numeric": schema.numeric_features, "categorical": schema.categorical_features},
        metrics=metrics,
        train_summary=train_summary,
        confidence_summary=confidence_summary,
    )


def evaluate_models_time_aware(
    df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    rs_model: Any,
    thickness_model: Any,
    rsu_model: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Simple time-aware evaluation:
    - sort by (target_id, lifetime)
    - use the last fraction of rows as test set
    """
    if df is None or df.empty:
        return {}

    test_fraction = float(config.get("model_settings", {}).get("test_fraction", 0.2))
    test_fraction = min(max(test_fraction, 0.05), 0.5)

    work = df.copy()
    if "target_id" in work.columns and "lifetime" in work.columns:
        # Keep original row index so we can align with X (built from the same df),
        # even when df is a filtered subset with non-consecutive index labels.
        work = work.sort_values(["target_id", "lifetime"])
    elif "lifetime" in work.columns:
        # Keep original row index for safe X alignment.
        work = work.sort_values(["lifetime"])

    n = len(work)
    split = int(np.floor((1.0 - test_fraction) * n))
    split = min(max(split, 1), n - 1)

    X_work = X.loc[work.index]
    X_train = X_work.iloc[:split]
    X_test = X_work.iloc[split:]

    metrics: dict[str, Any] = {"test_fraction": test_fraction, "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    def mae(model: Any, target_col: str) -> Optional[float]:
        if target_col not in work.columns:
            return None
        if model is None:
            return None
        y = pd.to_numeric(work[target_col], errors="coerce")
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]
        valid_mask = y_test.notna()
        if valid_mask.sum() == 0:
            return None
        y_pred = model.predict(X_test.loc[valid_mask.index])
        # Align to valid test rows
        y_pred = np.asarray(y_pred)[valid_mask.to_numpy()]
        return float(mean_absolute_error(y_test[valid_mask], y_pred))

    metrics["rs_mae"] = mae(rs_model, "rs")
    metrics["thickness_mae"] = mae(thickness_model, "thickness")
    metrics["rsu_mae"] = mae(rsu_model, "rsu")

    return metrics


def _train_regression_model(df: pd.DataFrame, X: pd.DataFrame, *, target_col: str, config: dict[str, Any]) -> Any:
    if df is None or df.empty or X is None or X.empty:
        return None

    feature_schema = get_feature_schema(config.get("feature_config", {}))
    numeric_features = [c for c in feature_schema.numeric_features if c in X.columns]
    categorical_features = [c for c in feature_schema.categorical_features if c in X.columns]

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]

    if len(y_valid) < 10:
        # Too little data for meaningful model; return None and let UI show low confidence.
        return None

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model_type = str(config.get("model_settings", {}).get(f"{target_col}_model", config.get("model_settings", {}).get("rs_model", "gbr")))
    random_state = int(config.get("model_settings", {}).get("random_state", 0))

    if model_type.lower() in {"rf", "random_forest"}:
        model = RandomForestRegressor(
            n_estimators=int(config.get("model_settings", {}).get("rf_n_estimators", 200)),
            random_state=random_state,
        )
    else:
        model = GradientBoostingRegressor(random_state=random_state)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_valid, y_valid)
    return pipeline


def _confidence_level(df: pd.DataFrame) -> str:
    n = len(df) if df is not None else 0
    if n < 10:
        return "Low"
    if n < 25:
        return "Medium"
    return "High"


def _confidence_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if df is None or df.empty:
        return ["no_data"]
    if len(df) < 10:
        notes.append("very_few_rows")
    if "target_id" in df.columns and df["target_id"].nunique() <= 1:
        notes.append("single_target_instance_only")
    if "lifetime" in df.columns:
        lifetime = pd.to_numeric(df["lifetime"], errors="coerce")
        if lifetime.notna().sum() > 0:
            span = float(lifetime.max() - lifetime.min())
            if span <= 0:
                notes.append("no_lifetime_span")
    return notes

