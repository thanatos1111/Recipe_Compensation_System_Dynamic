"""
Update pipeline for appending runs and refreshing artifacts (placeholder).
"""

from __future__ import annotations

from typing import Any

from core.schemas import MaterialDataset


def append_run_to_target_instance(
    material_dataset: MaterialDataset,
    target_id: str,
    new_row: dict[str, Any],
) -> MaterialDataset:
    """Append one new deposition run to the specified target instance."""
    raise NotImplementedError


def refresh_material_artifacts(material_dataset: MaterialDataset, train_config: dict[str, Any]) -> MaterialDataset:
    """Refresh material-level artifacts after data update."""
    raise NotImplementedError


def refresh_instance_artifacts(
    material_dataset: MaterialDataset,
    target_id: str,
    train_config: dict[str, Any],
) -> MaterialDataset:
    """Refresh instance-level artifacts after data update."""
    raise NotImplementedError

