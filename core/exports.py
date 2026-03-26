"""
Export utilities (placeholder).
"""

from __future__ import annotations

from typing import Any

from core.schemas import RecommendationArtifacts


def export_recommendation_table(
    artifacts: RecommendationArtifacts,
    out_path: str,
) -> None:
    """Export recommendation artifacts to disk (placeholder)."""
    raise NotImplementedError


def export_plots(out_path: str, *, plots: dict[str, Any]) -> None:
    """Export plots to disk (placeholder)."""
    raise NotImplementedError

