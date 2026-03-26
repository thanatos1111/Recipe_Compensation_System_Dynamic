"""
Input validation helpers (Milestone 2).

Validation focuses on producing clear warnings without crashing the app, so
the UI can still show partial results.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def validate_column_presence(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return warnings if required columns are missing."""
    missing = [c for c in required_columns if c not in df.columns]
    if not missing:
        return []
    return [f"missing_required_columns: {', '.join(missing)}"]


def validate_numeric_parsing(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Return warnings if numeric columns fail to parse."""
    warnings: list[str] = []

    for col in numeric_columns:
        if col not in df.columns:
            continue
        original_non_null = df[col].notna().sum()
        coerced = pd.to_numeric(df[col], errors="coerce")
        parsed_non_null = coerced.notna().sum()
        if original_non_null > 0 and parsed_non_null < original_non_null:
            bad_count = int(original_non_null - parsed_non_null)
            warnings.append(f"numeric_parsing_failed: {col} ({bad_count} rows)")

    return warnings


def validate_target_id_format(df: pd.DataFrame, *, material_name: str) -> list[str]:
    """Return warnings for missing/malformed target_id."""
    if "target_id" not in df.columns:
        return ["missing_required_columns: target_id"]

    warnings: list[str] = []
    target_id_series = df["target_id"].astype("string")
    target_id_stripped = target_id_series.str.strip()

    missing_mask = target_id_stripped.isna() | (target_id_stripped == "")
    missing_count = int(missing_mask.sum())

    # Recommended format: "<material>.<number>", e.g. "Ta.1"
    expected_re = re.compile(rf"^{re.escape(material_name)}\.\d+$")
    malformed_mask = (~missing_mask) & (~target_id_stripped.str.match(expected_re))
    malformed_count = int(malformed_mask.sum())

    if missing_count > 0 or malformed_count > 0:
        warnings.append(
            "target_id_missing_or_malformed: "
            f"missing={missing_count}, malformed={malformed_count}, expected_prefix={material_name}"
        )

    return warnings


def validate_dataset(df: pd.DataFrame, *, material_name: str, config: dict[str, Any]) -> list[str]:
    """Run all validation checks and return warnings."""
    warnings: list[str] = []

    # Canonical schema (Milestone 2 imports/normalization output).
    required_columns = [
        "lot_id",
        "date",
        "wafer_id",
        "lifetime",
        "incident_angle",
        "linear_offset",
        "rotations",
        "thickness",
        "rs",
        "rsu",
        "target_id",
        "ar_flow",
        "o2_flow",
    ]
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

    warnings.extend(validate_column_presence(df, required_columns))

    # Only run deeper checks if the required columns exist.
    if all(c in df.columns for c in required_columns):
        if len(df) < int(config.get("min_rows_warning", 10)):
            warnings.append(f"low_row_count: {len(df)}")

        warnings.extend(validate_numeric_parsing(df, numeric_columns))

        # Suspicious negatives: warn if any numeric feature is < 0.
        negative_cols: list[str] = []
        for col in numeric_columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any() and (coerced < 0).any():
                negative_cols.append(col)
        if negative_cols:
            warnings.append(f"suspicious_negative_values: {', '.join(negative_cols)}")

        # Duplicate lifetime rows within the same target instance.
        if "target_id" in df.columns and "lifetime" in df.columns:
            target_id_stripped = df["target_id"].astype("string").str.strip()
            valid_mask = target_id_stripped.notna() & (target_id_stripped != "") & df["lifetime"].notna()
            dup_mask = df.loc[valid_mask].duplicated(subset=["target_id", "lifetime"])
            if dup_mask.any():
                warnings.append("duplicate_lifetime_within_target_instance")

        warnings.extend(validate_target_id_format(df, material_name=material_name))

    return warnings

