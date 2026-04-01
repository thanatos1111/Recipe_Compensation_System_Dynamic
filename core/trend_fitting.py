"""
Baseline trend fitting (placeholder for Milestone 3).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def fit_parameter_trend(
    df: pd.DataFrame,
    *,
    parameter_name: str,
    method: str,
    step: float,
) -> Any:
    """Fit a baseline trend for one parameter vs lifetime."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "lifetime" not in df.columns or parameter_name not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if "in_spec" in work.columns:
        work = work[work["in_spec"] == True]  # noqa: E712
    work = work.dropna(subset=["lifetime", parameter_name])

    if work.empty:
        return pd.DataFrame()

    work = work.sort_values("lifetime")
    n = len(work)
    bin_count = int(min(10, max(1, np.sqrt(n))))

    lifetime_min = float(work["lifetime"].min())
    lifetime_max = float(work["lifetime"].max())

    if lifetime_min == lifetime_max:
        edges = np.array([lifetime_min, lifetime_max + 1e-9], dtype=float)
        bin_count = 1
    else:
        edges = np.linspace(lifetime_min, lifetime_max, bin_count + 1, dtype=float)

    labels = [f"bin_{i}" for i in range(bin_count)]
    work = work.copy()
    work["lifetime_bin"] = pd.cut(
        work["lifetime"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    grouped = work.groupby("lifetime_bin", observed=False)
    rows: list[dict[str, Any]] = []

    for bin_label, g in grouped:
        if g.empty:
            continue
        start = float(edges[int(str(bin_label).replace("bin_", ""))])
        end_idx = int(str(bin_label).replace("bin_", "")) + 1
        end = float(edges[min(end_idx, len(edges) - 1)])
        median_val = float(g[parameter_name].median())
        q = quantize_series_to_step(
            pd.Series([median_val]),
            step=step,
            # No clamp here: fit_parameter_trend does not know valid ranges.
            min_val=-1e308,
            max_val=1e308,
        ).iloc[0]
        rows.append(
            {
                "lifetime_bin_start": start,
                "lifetime_bin_end": end,
                f"{parameter_name}_ref": q,
                "count": int(len(g)),
                "method": method,
            }
        )

    return pd.DataFrame(rows)


def build_lifetime_reference_table(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create a simple lifetime-bin reference table from in-spec data."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "in_spec" in work.columns:
        in_spec_mask = work["in_spec"] == True  # noqa: E712
    else:
        in_spec_mask = pd.Series([True] * len(work), index=work.index)

    if "lifetime" not in work.columns:
        return pd.DataFrame()

    lifetime_min = float(work["lifetime"].min())
    lifetime_max = float(work["lifetime"].max())

    lifetime_bin_count = int(config.get("fit_settings", {}).get("lifetime_bin_count", 10))
    lifetime_bin_count = max(1, lifetime_bin_count)

    if lifetime_min == lifetime_max:
        edges = np.array([lifetime_min, lifetime_max + 1e-9], dtype=float)
        lifetime_bin_count = 1
    else:
        edges = np.linspace(lifetime_min, lifetime_max, lifetime_bin_count + 1, dtype=float)

    work["lifetime_bin"] = pd.cut(
        work["lifetime"],
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )

    parameter_constraints = config.get("parameter_constraints", {})

    def quantize_param(param: str, value: Optional[float]) -> Any:
        if value is None or pd.isna(value):
            return pd.NA
        pc = parameter_constraints.get(param, {})
        step = float(pc.get("step", 0.0))
        if step <= 0:
            # Defaults per technical spec request.
            step = 1.0 if param == "rotations" else 0.01
        if step <= 0:
            return float(value)

        # Only clamp when min/max look meaningful (avoid placeholder zeros).
        min_val = pc.get("min_value", None)
        max_val = pc.get("max_value", None)
        try:
            min_f = float(min_val) if min_val is not None else None
            max_f = float(max_val) if max_val is not None else None
        except (TypeError, ValueError):
            min_f = None
            max_f = None

        if (
            min_f is None
            or max_f is None
            or not np.isfinite(min_f)
            or not np.isfinite(max_f)
            or max_f <= min_f
        ):
            # Don't clamp; just quantize to step.
            min_f = -1e308
            max_f = 1e308

        return quantize_series_to_step(
            pd.Series([float(value)]),
            step=step,
            min_val=float(min_f),
            max_val=float(max_f),
        ).iloc[0]

    adjustable_params = ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"]
    output_cols = {
        "rs_pred": "rs",
        "thickness_pred": "thickness",
        "rsu_pred": "rsu",
    }

    rows: list[dict[str, Any]] = []
    total_bins = int(work["lifetime_bin"].max() + 1) if work["lifetime_bin"].notna().any() else 0
    for bin_idx in range(total_bins):
        bin_mask = work["lifetime_bin"] == bin_idx
        bin_total = work[bin_mask]
        if bin_total.empty:
            continue
        bin_in_spec = bin_total[in_spec_mask.loc[bin_mask]]

        in_spec_count = int(len(bin_in_spec))
        total_count = int(len(bin_total))
        in_spec_fraction = float(in_spec_count) / float(total_count) if total_count > 0 else pd.NA

        row: dict[str, Any] = {
            "lifetime_bin_start": float(edges[bin_idx]),
            "lifetime_bin_end": float(edges[bin_idx + 1]) if bin_idx + 1 < len(edges) else float(edges[-1]),
            "in_spec_count": in_spec_count,
            "total_count": total_count,
            "in_spec_fraction": in_spec_fraction,
        }

        # Recipe parameter reference values (median of in-spec rows).
        for p in adjustable_params:
            if p in bin_in_spec.columns:
                med = None if bin_in_spec.empty else float(bin_in_spec[p].median())
                row[f"{p}_rec"] = quantize_param(p, med)
            else:
                row[f"{p}_rec"] = pd.NA

        # Output reference predictions (median of in-spec rows).
        for out_name, col in output_cols.items():
            if col in bin_in_spec.columns and not bin_in_spec.empty:
                row[out_name] = float(bin_in_spec[col].median())
            else:
                row[out_name] = pd.NA

        rows.append(row)

    return pd.DataFrame(rows)


def quantize_series_to_step(
    values: pd.Series,
    *,
    step: float,
    min_val: float,
    max_val: float,
) -> pd.Series:
    """Quantize numeric values to valid step size."""
    if values is None:
        return pd.Series(dtype=float)

    if step is None or float(step) <= 0:
        return pd.to_numeric(values, errors="coerce")

    x = pd.to_numeric(values, errors="coerce").astype(float)
    q = np.round(x / float(step)) * float(step)

    if min_val is not None and np.isfinite(float(min_val)):
        q = np.maximum(q, float(min_val))
    if max_val is not None and np.isfinite(float(max_val)):
        q = np.minimum(q, float(max_val))

    return pd.Series(q, index=values.index)

