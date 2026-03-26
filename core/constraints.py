"""
Constraint checks for discrete recipe optimization (placeholder).
"""

from __future__ import annotations

from typing import Any


def validate_candidate_parameters(candidate: dict[str, Any], parameter_config: dict[str, Any]) -> list[str]:
    """Return a list of constraint violation messages."""
    raise NotImplementedError


def enforce_discrete_step(candidate: dict[str, Any], parameter_config: dict[str, Any]) -> dict[str, Any]:
    """Quantize candidate parameters to valid discrete step sizes."""
    raise NotImplementedError


def enforce_coupling_rules(candidate: dict[str, Any], coupling_config: dict[str, Any]) -> dict[str, Any]:
    """Enforce coupling rules like total flow constancy."""
    raise NotImplementedError

