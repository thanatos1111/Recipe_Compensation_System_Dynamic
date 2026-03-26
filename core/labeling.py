"""
Spec labeling and derived-feature computation (placeholder).

Milestone 3 will implement in-spec / out-of-spec rules and derived columns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.schemas import SpecConfig


def compute_derived_features(df: pd.DataFrame, spec_config: SpecConfig) -> pd.DataFrame:
    """Compute derived columns used for labeling and later plotting/modeling."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure stable ordering for delta computations.
    if "target_id" in out.columns and "lifetime" in out.columns:
        out = out.sort_values(["target_id", "lifetime"]).reset_index(drop=True)

    # Total gas flow + O2 ratio.
    if {"ar_flow", "o2_flow"}.issubset(out.columns):
        out["total_flow"] = out["ar_flow"] + out["o2_flow"]
        out["o2_ratio"] = out["o2_flow"] / out["total_flow"]
        out.loc[out["total_flow"] <= 0, "o2_ratio"] = pd.NA
    else:
        out["total_flow"] = pd.NA
        out["o2_ratio"] = pd.NA

    # RS derived features (for explainability).
    if spec_config.use_rs_target_mode and spec_config.rs_target is not None and "rs" in out.columns:
        out["rs_error"] = out["rs"] - float(spec_config.rs_target)
        out["abs_rs_error"] = out["rs_error"].abs()
    else:
        out["rs_error"] = pd.NA
        out["abs_rs_error"] = pd.NA

    # RSU margin (for explainability).
    if "rsu" in out.columns and spec_config.rsu_max is not None and spec_config.rsu_max > 0:
        out["rsu_margin"] = float(spec_config.rsu_max) - out["rsu"]
    else:
        out["rsu_margin"] = pd.NA

    # Thickness derived features (for explainability).
    if (
        spec_config.use_thickness_target_mode
        and spec_config.thickness_target is not None
        and "thickness" in out.columns
    ):
        out["thickness_error"] = out["thickness"] - float(spec_config.thickness_target)
    else:
        out["thickness_error"] = pd.NA

    # Delta recipe parameter values within each target instance.
    delta_params = ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"]
    if "target_id" in out.columns and "lifetime" in out.columns:
        for param in delta_params:
            if param in out.columns:
                out[f"delta_{param}"] = out.groupby("target_id")[param].diff()
    else:
        for param in delta_params:
            out[f"delta_{param}"] = pd.NA

    # Apply in-spec/out-of-spec label.
    out = apply_spec_labels(out, spec_config)
    return out


def apply_spec_labels(df: pd.DataFrame, spec_config: SpecConfig) -> pd.DataFrame:
    """Apply in-spec / out-of-spec labels based on spec_config."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # RSU criterion.
    if not spec_config.use_rsu_spec or "rsu" not in out.columns or spec_config.rsu_max is None or spec_config.rsu_max <= 0:
        rsu_ok = True
    else:
        rsu_ok = out["rsu"] <= float(spec_config.rsu_max)

    # RS criterion: target-mode or range-mode.
    if not spec_config.use_rs_spec or "rs" not in out.columns:
        rs_ok = True
    elif spec_config.use_rs_target_mode:
        if spec_config.rs_target is None or spec_config.rs_tol is None:
            rs_ok = True
        else:
            rs_ok = (out["rs"] - float(spec_config.rs_target)).abs() <= float(spec_config.rs_tol)
    else:
        # Range-mode: allow partial bounds when one side is "unset".
        if spec_config.rs_min is None and spec_config.rs_max is None:
            rs_ok = True
        elif spec_config.rs_min is None:
            rs_ok = out["rs"] <= float(spec_config.rs_max)  # type: ignore[arg-type]
        elif spec_config.rs_max is None:
            rs_ok = out["rs"] >= float(spec_config.rs_min)  # type: ignore[arg-type]
        else:
            rs_ok = (out["rs"] >= float(spec_config.rs_min)) & (out["rs"] <= float(spec_config.rs_max))  # type: ignore[arg-type]

    # Thickness criterion: target-mode or range-mode.
    if not spec_config.use_thickness_spec or "thickness" not in out.columns:
        thickness_ok = True
    elif spec_config.use_thickness_target_mode:
        if spec_config.thickness_target is None or spec_config.thickness_tol is None:
            thickness_ok = True
        else:
            thickness_ok = (out["thickness"] - float(spec_config.thickness_target)).abs() <= float(
                spec_config.thickness_tol
            )
    else:
        # Range-mode: allow partial bounds when one side is "unset".
        if spec_config.thickness_min is None and spec_config.thickness_max is None:
            thickness_ok = True
        elif spec_config.thickness_min is None:
            thickness_ok = out["thickness"] <= float(spec_config.thickness_max)  # type: ignore[arg-type]
        elif spec_config.thickness_max is None:
            thickness_ok = out["thickness"] >= float(spec_config.thickness_min)  # type: ignore[arg-type]
        else:
            thickness_ok = (out["thickness"] >= float(spec_config.thickness_min)) & (
                out["thickness"] <= float(spec_config.thickness_max)
            )  # type: ignore[arg-type]

    out["in_spec"] = rsu_ok & rs_ok & thickness_ok
    return out

