"""
Read-only pipeline inspection helpers for model-area data processing view.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.feature_engineering import build_feature_matrix, get_feature_schema


def build_processing_view_data(
    *,
    material_dataset: Any,
    active_target_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build stage-by-stage dataframes and summaries for one material.

    Notes:
    - Different materials are not mixed: this consumes one MaterialDataset only.
    - "raw rows" are represented by current material rows before additional filtering.
    """
    all_df = material_dataset.all_records.copy() if material_dataset is not None and material_dataset.all_records is not None else pd.DataFrame()
    target_instances = getattr(material_dataset, "target_instances", {}) or {}
    active_df = target_instances.get(active_target_id).records.copy() if active_target_id in target_instances else pd.DataFrame()

    feature_config = config.get("feature_config", {}) if isinstance(config, dict) else {}
    feature_schema = get_feature_schema(feature_config)
    feature_list = [c for c in feature_schema.numeric_features + feature_schema.categorical_features]

    normalized_df = all_df.copy()
    grouped_df = _grouped_summary_df(target_instances)
    derived_cols = [c for c in ["total_flow", "o2_ratio", "lifetime_used", "lifetime_end", "rs_error", "abs_rs_error", "rsu_margin", "thickness_error"] if c in all_df.columns]
    derived_df = all_df[["target_id", "lifetime"] + derived_cols].copy() if derived_cols else pd.DataFrame()
    labeled_df = all_df.copy() if "in_spec" in all_df.columns else pd.DataFrame()
    trend_df = _trend_rows(active_df)
    training_df = all_df.copy()
    excluded_df = _excluded_rows(all_df, active_target_id, feature_list)

    stages = [
        ("raw_rows", "Raw rows", all_df),
        ("normalized_rows", "Normalized rows", normalized_df),
        ("grouped_target_instances", "Grouped target instances", grouped_df),
        ("derived_columns", "Derived columns", derived_df),
        ("in_spec_labels", "In-spec labels", labeled_df),
        ("trend_fitting_rows", "Rows used for trend fitting", trend_df),
        ("training_rows", "Rows used for training", training_df),
        ("excluded_rows", "Excluded rows and reasons", excluded_df),
    ]

    stage_payload: list[dict[str, Any]] = []
    for stage_id, stage_name, sdf in stages:
        sdf = sdf if isinstance(sdf, pd.DataFrame) else pd.DataFrame()
        stage_payload.append(
            {
                "id": stage_id,
                "name": stage_name,
                "df": sdf,
                "summary": _summarize_df(sdf),
            }
        )

    return {
        "stages": stage_payload,
        "feature_list": feature_list,
    }


def _grouped_summary_df(target_instances: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_id, inst in (target_instances or {}).items():
        records = inst.records if hasattr(inst, "records") else None
        count = int(len(records)) if records is not None else 0
        lt_min = None
        lt_max = None
        if records is not None and not records.empty and "lifetime" in records.columns:
            lt = pd.to_numeric(records["lifetime"], errors="coerce").dropna()
            if not lt.empty:
                lt_min = float(lt.min())
                lt_max = float(lt.max())
        rows.append({"target_id": str(target_id), "row_count": count, "lifetime_min": lt_min, "lifetime_max": lt_max})
    return pd.DataFrame(rows)


def _trend_rows(active_df: pd.DataFrame) -> pd.DataFrame:
    if active_df is None or active_df.empty:
        return pd.DataFrame()
    out = active_df.copy()
    if "in_spec" in out.columns:
        out = out[out["in_spec"] == True]  # noqa: E712
    needed = [c for c in ["lifetime", "incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"] if c in out.columns]
    if needed:
        out = out.dropna(subset=[c for c in ["lifetime"] if c in needed])
    return out.reset_index(drop=True)


def _excluded_rows(df: pd.DataFrame, active_target_id: str, feature_list: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    reasons: list[str] = []
    for _, row in work.iterrows():
        r: list[str] = []
        if pd.isna(row.get("target_id")) or str(row.get("target_id", "")).strip() == "":
            r.append("missing_target_id")
        if pd.isna(row.get("lifetime")):
            r.append("missing_lifetime")
        if str(row.get("target_id", "")) == str(active_target_id) and "in_spec" in work.columns and pd.isna(row.get("in_spec")):
            r.append("missing_in_spec_label")
        missing_feats = [f for f in feature_list if f in work.columns and pd.isna(row.get(f))]
        if missing_feats:
            r.append(f"missing_features:{','.join(missing_feats[:4])}")
        reasons.append(";".join(r))
    work["exclude_reason"] = reasons
    out = work[work["exclude_reason"].astype(str) != ""].copy()
    return out.reset_index(drop=True)


def _summarize_df(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "row_count": 0,
            "missing_values": 0,
            "in_spec_count": 0,
            "target_instance_count": 0,
        }
    missing_values = int(df.isna().sum().sum())
    in_spec_count = int(df["in_spec"].sum()) if "in_spec" in df.columns else 0
    target_count = int(df["target_id"].nunique()) if "target_id" in df.columns else 0
    return {
        "row_count": int(len(df)),
        "missing_values": missing_values,
        "in_spec_count": in_spec_count,
        "target_instance_count": target_count,
    }
