"""
Feature engineering (placeholder for Milestone 4).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_feature_matrix(df: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Build a model feature matrix from canonical material dataframe."""
    raise NotImplementedError


def encode_target_identity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Encode or represent target identity while preserving traceability."""
    raise NotImplementedError


def build_instance_correction_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Build inputs for instance-level correction layer."""
    raise NotImplementedError

