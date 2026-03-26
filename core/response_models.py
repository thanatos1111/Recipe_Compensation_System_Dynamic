"""
Material-specific response model training + prediction (placeholder).

Milestone 4 will implement model training and evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def train_rs_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RS regression model."""
    raise NotImplementedError


def train_thickness_model(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Train a thickness regression model."""
    raise NotImplementedError


def train_rsu_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RSU regression model."""
    raise NotImplementedError


def train_spec_classifier(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Optional classifier for in-spec probability."""
    raise NotImplementedError


def predict_outputs(model_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Predict RS, thickness, and RSU (and optional in-spec probability)."""
    raise NotImplementedError

