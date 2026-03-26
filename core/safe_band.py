"""
Feasible-region / safe-band estimation (placeholder).
"""

from __future__ import annotations

from typing import Any


def estimate_parameter_band(
    center_recipe: dict[str, Any],
    model_bundle: dict[str, Any],
    *,
    parameter_name: str,
    spec_config: Any,
    step: float,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Estimate acceptable parameter interval around a center recipe."""
    raise NotImplementedError


def build_feasible_map(
    center_recipe: dict[str, Any],
    model_bundle: dict[str, Any],
    varying_parameters: list[str],
    *,
    spec_config: Any,
) -> dict[str, Any]:
    """Build a feasible map (placeholder)."""
    raise NotImplementedError

