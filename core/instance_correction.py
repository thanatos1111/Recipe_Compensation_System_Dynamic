"""
Instance correction layer (placeholder for Milestone 4).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def fit_instance_bias(
    material_model_bundle: dict[str, Any],
    instance_df: pd.DataFrame,
) -> Any:
    """Estimate instance-specific residual bias terms."""
    raise NotImplementedError


def apply_instance_correction(predictions: dict[str, Any], correction_artifacts: Any) -> dict[str, Any]:
    """Apply correction artifacts to model predictions."""
    raise NotImplementedError


def summarize_recent_residuals(instance_df: pd.DataFrame, material_model_bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize residual drift for the active target instance."""
    raise NotImplementedError

