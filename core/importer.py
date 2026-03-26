"""
Excel workbook import + normalization (Milestone 2).

Responsibilities:
- Load an .xlsx workbook
- Treat each sheet as one material
- Normalize columns using configurable raw->canonical column mapping
- Group rows into target instances by `target_id`
- Produce warnings via `core.validation`
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.schemas import MaterialDataset, TargetInstance
from core.validation import validate_dataset


def load_workbook(path: str) -> dict[str, pd.DataFrame]:
    """Load a single .xlsx workbook into a dict of sheet->DataFrame."""
    # `sheet_name=None` loads all sheets at once.
    # `openpyxl` engine is required for .xlsx.
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")


def normalize_material_sheet(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    config: dict[str, Any],
    material_name: str,
) -> pd.DataFrame:
    """
    Normalize one material sheet into canonical columns.

    The importer does not assume a fixed column order; it uses `column_map`
    from config to map raw Excel headers to canonical field names.
    """
    del config  # config reserved for future normalization policy switches

    def normalize_header(value: Any) -> str:
        """
        Normalize an Excel header for robust matching.

        Strategy:
        - collapse whitespace
        - trim + lowercase
        - keep a whitespace-tolerant form for exact-phrase matching
        """
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip().lower()

    def normalize_header_compact(value: Any) -> str:
        """
        More aggressive normalization for cases like:
        - "IncidentAngle" vs "Incident Angle"
        - "Linear-Offset" vs "Linear Offset"
        - hidden punctuation/underscore differences
        """
        s = normalize_header(value)
        return re.sub(r"[^a-z0-9]+", "", s)

    # Build a normalized header->actual-header mapping from the raw dataframe.
    # This handles cases like "Ar " vs "Ar" and "incident angle" vs "Incident Angle".
    raw_header_map: dict[str, str] = {}
    raw_header_map_compact: dict[str, str] = {}
    for col in df.columns:
        norm = normalize_header(col)
        norm_compact = normalize_header_compact(col)
        if norm:
            raw_header_map[norm] = str(col)
        if norm_compact:
            raw_header_map_compact[norm_compact] = str(col)

    out = pd.DataFrame()

    # Map raw columns (Excel headers) to canonical columns.
    for raw_col, canonical_col in column_map.items():
        norm_raw = normalize_header(raw_col)
        actual_col = raw_header_map.get(norm_raw)
        if actual_col is None:
            actual_col = raw_header_map_compact.get(normalize_header_compact(raw_col))
        if actual_col is not None and actual_col in df.columns:
            out[canonical_col] = df[actual_col]
        else:
            # Missing columns are handled as warnings later.
            out[canonical_col] = pd.NA

    # Heuristic fallbacks for commonly-extended columns not present in mapping.
    # This helps when headers look like "rpm(rotation velocity)" or "Power(W)".
    def _first_matching_col(tokens: list[str]) -> str | None:
        # Try compact match by containment.
        for col in df.columns:
            col_compact = normalize_header_compact(col)
            for t in tokens:
                if t and (t in col_compact):
                    return str(col)
        return None

    if "rpm" not in out.columns or out["rpm"].isna().all():
        c = _first_matching_col(["rpm"])
        if c is not None:
            out["rpm"] = df[c]

    if "power" not in out.columns or out["power"].isna().all():
        c = _first_matching_col(["power", "kw", "watt"])
        if c is not None:
            out["power"] = df[c]

    # Required canonical derived context.
    out["material_name"] = material_name

    # Normalize/parse common canonical types.
    if "target_id" in out.columns:
        out["target_id"] = out["target_id"].astype("string").str.strip()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    numeric_columns = [
        "lifetime",
        "incident_angle",
        "linear_offset",
        "rotations",
        "rpm",
        "power",
        "thickness",
        "rs",
        "rsu",
        "ar_flow",
        "o2_flow",
    ]
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Ensure a stable column set/order for downstream panels.
    preferred_columns = [
        "material_name",
        "lot_id",
        "date",
        "wafer_id",
        "lifetime",
        "ar_flow",
        "o2_flow",
        "incident_angle",
        "linear_offset",
        "rotations",
        "rpm",
        "power",
        "thickness",
        "rs",
        "rsu",
        "target_id",
    ]
    for c in preferred_columns:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[preferred_columns]

    return out


def group_target_instances(df: pd.DataFrame) -> dict[str, TargetInstance]:
    """Group canonical rows by target_id into TargetInstance objects."""
    if df.empty:
        return {}
    if "target_id" not in df.columns:
        return {}

    material_name = ""
    if "material_name" in df.columns and df["material_name"].notna().any():
        material_name = str(df["material_name"].dropna().iloc[0])

    df_valid = df.copy()
    target_id_stripped = df_valid["target_id"].astype("string").str.strip()
    df_valid = df_valid[target_id_stripped.notna() & (target_id_stripped != "")]
    df_valid = df_valid.assign(target_id=target_id_stripped)

    target_instances: dict[str, TargetInstance] = {}
    for target_id, group_df in df_valid.groupby("target_id"):
        sorted_df = group_df.sort_values("lifetime") if "lifetime" in group_df.columns else group_df
        target_instances[str(target_id)] = TargetInstance(
            target_id=str(target_id),
            material_name=material_name,
            records=sorted_df.reset_index(drop=True),
            status="historical",
            fit_artifacts=None,
            instance_correction_artifacts=None,
            warnings=[],
        )

    return target_instances


def build_material_dataset(
    sheet_name: str,
    df: pd.DataFrame,
    *,
    column_map: dict[str, Any],
    config: dict[str, Any],
) -> MaterialDataset:
    """Build a MaterialDataset for one sheet/material."""
    normalized = normalize_material_sheet(df, column_map=column_map, config=config, material_name=sheet_name)

    warnings = validate_dataset(normalized, material_name=sheet_name, config=config)
    target_instances = group_target_instances(normalized)

    return MaterialDataset(
        material_name=sheet_name,
        source_sheet_name=sheet_name,
        all_records=normalized,
        target_instances=target_instances,
        column_map=column_map,
        # SpecConfig/Model artifacts are milestone-1 scaffolding defaults.
        warnings=warnings,
    )

