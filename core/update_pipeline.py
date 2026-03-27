"""
Update pipeline for appending runs and refreshing artifacts (placeholder).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.labeling import compute_derived_features
from core.response_models import train_material_models
from core.schemas import MaterialDataset, TargetInstance
from core.instance_correction import fit_instance_bias


def append_run_to_target_instance(
    material_dataset: MaterialDataset,
    target_id: str,
    new_row: dict[str, Any],
) -> MaterialDataset:
    """Append one new deposition run to the specified target instance."""
    if material_dataset is None:
        return material_dataset
    if target_id not in material_dataset.target_instances:
        # If unknown target instance, create it so the append still works.
        material_dataset.target_instances[target_id] = TargetInstance(
            target_id=target_id,
            material_name=material_dataset.material_name,
        )

    inst = material_dataset.target_instances[target_id]
    new_df = pd.DataFrame([new_row])

    if inst.records is None or inst.records.empty:
        inst.records = new_df
    else:
        inst.records = pd.concat([inst.records, new_df], ignore_index=True)

    # Sort by lifetime (starting lifetime) for stable delta computation later.
    if "lifetime" in inst.records.columns:
        inst.records = inst.records.sort_values("lifetime").reset_index(drop=True)
    return material_dataset


def refresh_material_artifacts(material_dataset: MaterialDataset, train_config: dict[str, Any]) -> MaterialDataset:
    """Refresh material-level artifacts after data update."""
    df = material_dataset.all_records
    if df is None or df.empty:
        material_dataset.material_model_artifacts = material_dataset.material_model_artifacts
        return material_dataset

    material_artifacts = train_material_models(df, config=train_config)
    material_dataset.material_model_artifacts = material_artifacts
    return material_dataset


def refresh_instance_artifacts(
    material_dataset: MaterialDataset,
    target_id: str,
    train_config: dict[str, Any],
) -> MaterialDataset:
    """Refresh instance-level artifacts after data update."""
    if target_id not in material_dataset.target_instances:
        return material_dataset

    inst = material_dataset.target_instances[target_id]
    if inst.records is None or inst.records.empty:
        inst.instance_correction_artifacts = None
        return material_dataset

    model_bundle = {
        "rs_model": material_dataset.material_model_artifacts.rs_model,
        "thickness_model": material_dataset.material_model_artifacts.thickness_model,
        "rsu_model": material_dataset.material_model_artifacts.rsu_model,
        "feature_config": train_config.get("feature_config", {}),
    }
    correction = fit_instance_bias(model_bundle, inst.records)
    inst.instance_correction_artifacts = correction
    return material_dataset


def append_runs_and_refresh(
    material_dataset: MaterialDataset,
    target_id: str,
    new_rows_df: pd.DataFrame,
    *,
    train_config: dict[str, Any],
    spec_config: Optional[Any] = None,
) -> MaterialDataset:
    """
    Append multiple new rows to one target instance, then recompute:
    - derived columns + `in_spec`
    - material-level model artifacts
    - instance-level correction artifacts
    """
    if material_dataset is None:
        return material_dataset
    if new_rows_df is None or new_rows_df.empty:
        return material_dataset

    # Append to target instance records (keep rows as canonical schema).
    if target_id not in material_dataset.target_instances:
        material_dataset.target_instances[target_id] = TargetInstance(
            target_id=target_id,
            material_name=material_dataset.material_name,
        )
    inst = material_dataset.target_instances[target_id]

    if inst.records is None or inst.records.empty:
        inst.records = new_rows_df.reset_index(drop=True)
    else:
        inst.records = pd.concat([inst.records, new_rows_df], ignore_index=True)

    if "lifetime" in inst.records.columns:
        inst.records = inst.records.sort_values("lifetime").reset_index(drop=True)

    # Rebuild material-level dataframe from instances (preserve target_id identity).
    all_records = []
    for _tid, _inst in material_dataset.target_instances.items():
        if _inst.records is not None and not _inst.records.empty:
            all_records.append(_inst.records)
    material_dataset.all_records = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()

    # Use provided spec_config if given; otherwise material_dataset.spec_config.
    used_spec = spec_config if spec_config is not None else material_dataset.spec_config
    material_dataset.spec_config = used_spec

    labeled = compute_derived_features(material_dataset.all_records, material_dataset.spec_config)
    material_dataset.all_records = labeled

    # Update each instance's records view from the labeled material table.
    if "target_id" in labeled.columns:
        for tid, iinst in material_dataset.target_instances.items():
            subset = labeled[labeled["target_id"] == tid].copy()
            if "lifetime" in subset.columns:
                subset = subset.sort_values("lifetime").reset_index(drop=True)
            iinst.records = subset

    # Refresh artifacts.
    refresh_material_artifacts(material_dataset, train_config=train_config)
    refresh_instance_artifacts(material_dataset, target_id=target_id, train_config=train_config)

    # Clear previous recommendation artifacts; UI can regenerate.
    material_dataset.recommendation_artifacts = None

    return material_dataset

