"""
Optimizer + scoring (placeholder for Milestone 5).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def score_candidate(
    candidate: dict[str, Any],
    predicted_outputs: dict[str, Any],
    spec_config: Any,
    weights: dict[str, float],
    reference_recipe: dict[str, Any],
) -> dict[str, Any]:
    """Compute score breakdown for one candidate."""
    raise NotImplementedError


def rank_candidates(
    candidates: pd.DataFrame,
    model_bundle: dict[str, Any],
    context: dict[str, Any],
) -> pd.DataFrame:
    """Predict, score, and rank candidates."""
    raise NotImplementedError


def select_best_candidate(ranked_df: pd.DataFrame) -> dict[str, Any]:
    """Select the best candidate (top row) from ranked candidates."""
    raise NotImplementedError

