"""
Goal preset helpers for the auto-decision UI.

This module is UI-agnostic: it only defines preset mappings and normalization so
tests can validate behavior without requiring Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AutoDecisionGoalPreset:
    key: str
    display_name: str
    description: str
    ranking_objective: str
    uncertainty_mode: str
    # Whether to enable conformal interval metrics during benchmarking.
    uncertainty_enabled: bool = False
    conformal_alpha: float = 0.1
    calibration_fraction: float = 0.2
    # Optional weighted_combined weights.
    ranking_weights: Optional[dict[str, float]] = None
    # Suggested shortlist size for optional backtest.
    shortlist_size: int = 3


def get_auto_decision_goal_presets() -> list[AutoDecisionGoalPreset]:
    return [
        AutoDecisionGoalPreset(
            key="balanced",
            display_name="Balanced",
            description="Use the same ranking objective as manual benchmark defaults, with conservative uncertainty handling.",
            ranking_objective="spec_pass_first",
            uncertainty_mode="warn_only",
            uncertainty_enabled=False,
            shortlist_size=3,
        ),
        AutoDecisionGoalPreset(
            key="prediction_first",
            display_name="Prediction-first",
            description="Prioritize prediction accuracy first (RS MAE), then spec-pass accuracy.",
            ranking_objective="rs_first",
            uncertainty_mode="ignore",
            uncertainty_enabled=False,
            shortlist_size=3,
        ),
        AutoDecisionGoalPreset(
            key="recommendation_first",
            display_name="Recommendation-first",
            description="Prioritize spec-pass first and use a larger shortlist so backtest can break ties when feasible.",
            ranking_objective="spec_pass_first",
            uncertainty_mode="ignore",
            uncertainty_enabled=False,
            shortlist_size=5,
        ),
        AutoDecisionGoalPreset(
            key="conservative_safe",
            display_name="Conservative-safe",
            description="Prefer stable ranking and warn on uncertainty miscalibration (does not change rank).",
            ranking_objective="spec_pass_first",
            uncertainty_mode="warn_only",
            uncertainty_enabled=True,
            conformal_alpha=0.1,
            calibration_fraction=0.2,
            shortlist_size=3,
        ),
        AutoDecisionGoalPreset(
            key="weighted_balanced",
            display_name="Weighted combined (balanced)",
            description="Use weighted_combined with equal weights across summary metrics.",
            ranking_objective="weighted_combined",
            uncertainty_mode="ignore",
            uncertainty_enabled=False,
            ranking_weights={
                "spec_pass_accuracy": 1.0,
                "rs_mae": 1.0,
                "thickness_mae": 1.0,
                "rsu_mae": 1.0,
            },
            shortlist_size=3,
        ),
    ]


def find_goal_preset(key: str) -> Optional[AutoDecisionGoalPreset]:
    k = str(key or "").strip()
    for p in get_auto_decision_goal_presets():
        if p.key == k:
            return p
    return None


def apply_goal_preset_to_settings(
    preset_key: str,
    *,
    current_settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Return a normalized settings dict after applying a preset.

    Shape matches how `ui/model_panel.py` assembles auto-decision config.
    """
    base: dict[str, Any] = dict(current_settings or {})
    p = find_goal_preset(preset_key)
    if p is None:
        return base

    base["ranking_objective"] = p.ranking_objective
    base["uncertainty_mode"] = p.uncertainty_mode
    base["uncertainty_enabled"] = bool(p.uncertainty_enabled)
    base["conformal_alpha"] = float(p.conformal_alpha)
    base["calibration_fraction"] = float(p.calibration_fraction)
    base["ranking_weights"] = dict(p.ranking_weights or {}) if p.ranking_weights is not None else {}
    base["shortlist_size"] = int(p.shortlist_size)
    return base

