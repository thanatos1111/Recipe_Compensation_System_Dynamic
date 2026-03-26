"""
Feature engineering (placeholder for Milestone 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    numeric_features: list[str]
    categorical_features: list[str]


def build_feature_matrix(df: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Build a model feature matrix from canonical material dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Default feature set per design spec.
    default_numeric = [
        "lifetime",
        "incident_angle",
        "linear_offset",
        "rotations",
        "ar_flow",
        "o2_flow",
        "o2_ratio",
        "total_flow",
    ]
    default_categorical = ["target_id"]

    numeric_features = list(feature_config.get("numeric_features", default_numeric))
    categorical_features = list(feature_config.get("categorical_features", default_categorical))

    cols: list[str] = []
    for c in numeric_features + categorical_features:
        if c not in cols:
            cols.append(c)

    X = df.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[cols]

    # Ensure categorical fields are strings for encoding.
    for c in categorical_features:
        X[c] = X[c].astype("string")

    return X


def encode_target_identity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Encode or represent target identity while preserving traceability."""
    # Milestone 4 default: keep target_id as a categorical feature for OneHotEncoder.
    # This function exists for future strategies (e.g., grouped correction).
    if df is None or df.empty:
        return df
    if mode not in {"onehot", "raw"}:
        return df
    if "target_id" in df.columns:
        df = df.copy()
        df["target_id"] = df["target_id"].astype("string")
    return df


def build_instance_correction_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Build inputs for instance-level correction layer."""
    if df is None or df.empty:
        return pd.DataFrame()
    # Keep minimal: correction uses recent residuals vs lifetime.
    cols = [c for c in ["lifetime", "rs", "thickness", "rsu"] if c in df.columns]
    return df[cols].copy()


def get_feature_schema(feature_config: dict[str, Any]) -> FeatureSchema:
    default_numeric = [
        "lifetime",
        "lifetime_used",
        "lifetime_end",
        "incident_angle",
        "linear_offset",
        "rotations",
        "rpm",
        "power",
        "ar_flow",
        "o2_flow",
        "o2_ratio",
        "total_flow",
    ]
    default_categorical = ["target_id"]
    numeric_features = list(feature_config.get("numeric_features", default_numeric))
    categorical_features = list(feature_config.get("categorical_features", default_categorical))
    return FeatureSchema(numeric_features=numeric_features, categorical_features=categorical_features)

