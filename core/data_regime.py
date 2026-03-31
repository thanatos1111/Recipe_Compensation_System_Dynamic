"""
Data regime inspection helpers for auto-decision.

The auto-decision engine must remain material-local:
- each sheet is one material
- rows are grouped by Target ID into target instances
- datasets from different materials must not be mixed

This module provides lightweight metadata that downstream planners can use to
select valid split modes and evaluation stages without running training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class DataRegime:
    """Material-local data regime summary."""

    total_rows: int
    target_count: int
    rows_per_target: dict[str, int]

    leave_one_target_out_feasible: bool
    active_target_cutoff_feasible: bool
    uncertainty_calibration_feasible: bool
    recommendation_backtest_feasible: bool

    regime_label: str
    notes: list[str]


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def inspect_data_regime(df: pd.DataFrame, config: dict[str, Any]) -> DataRegime:
    """
    Inspect a *single material* dataframe and return data regime metadata.

    Parameters
    ----------
    df:
        Canonical material-local dataframe (must include `target_id` for most checks).
    config:
        Effective config mapping. Uses:
        - ``benchmark_settings`` for split thresholds and uncertainty config
        - ``auto_decision`` for active_target_id/cutoff hints (optional)

    Notes
    -----
    This function does not train models. It only inspects dataset shape and
    config thresholds to decide which evaluation paths are valid.
    """
    if df is None or df.empty:
        return DataRegime(
            total_rows=0,
            target_count=0,
            rows_per_target={},
            leave_one_target_out_feasible=False,
            active_target_cutoff_feasible=False,
            uncertainty_calibration_feasible=False,
            recommendation_backtest_feasible=False,
            regime_label="very_sparse",
            notes=["empty_dataset"],
        )

    notes: list[str] = []
    total_rows = int(len(df))
    if "target_id" not in df.columns:
        return DataRegime(
            total_rows=total_rows,
            target_count=0,
            rows_per_target={},
            leave_one_target_out_feasible=False,
            active_target_cutoff_feasible=False,
            uncertainty_calibration_feasible=False,
            recommendation_backtest_feasible=False,
            regime_label="very_sparse",
            notes=["missing_target_id_column"],
        )

    work = df.copy()
    work["target_id"] = work["target_id"].astype("string").str.strip()
    work = work[work["target_id"].notna() & (work["target_id"] != "")]

    rows_per_target: dict[str, int] = {str(t): int(n) for t, n in work["target_id"].value_counts().items()}
    target_count = int(len(rows_per_target))

    bs = config.get("benchmark_settings") or {}
    lo = dict(bs.get("leave_one_target_out", {}) or {})
    lo_min_train = _safe_int(lo.get("min_train_rows", 10), 10)
    lo_min_test = _safe_int(lo.get("min_test_rows", 3), 3)

    ac = dict(bs.get("active_target_cutoff", {}) or {})
    aid = str(ac.get("active_target_id") or (config.get("auto_decision", {}) or {}).get("active_target_id") or "").strip()
    cutoff = ac.get("cutoff_lifetime")
    cutoff_f = _safe_float(cutoff, default=float("nan")) if cutoff is not None else float("nan")
    history_ids = tuple(ac.get("history_target_ids") or ())

    # Leave-one-target-out feasibility:
    # at least 2 targets and at least one fold with adequate train/test sizes.
    loto_feasible = False
    if target_count >= 2:
        for tid, n_test in rows_per_target.items():
            n_train = total_rows - int(n_test)
            if n_train >= lo_min_train and n_test >= lo_min_test:
                loto_feasible = True
                break
    if not loto_feasible:
        notes.append("leave_one_target_out_not_feasible")

    # Active-target-cutoff feasibility:
    # requires active_target_id + numeric cutoff, plus non-empty train/test partitions.
    active_cutoff_feasible = False
    if aid and "lifetime" in work.columns and pd.notna(cutoff_f):
        try:
            lt = pd.to_numeric(work["lifetime"], errors="coerce")
            active_rows = work[work["target_id"].astype(str) == aid].copy()
            if not active_rows.empty:
                active_rows = active_rows.assign(lifetime=lt.loc[active_rows.index])
                train_active = active_rows[active_rows["lifetime"] <= float(cutoff_f)]
                test_active = active_rows[active_rows["lifetime"] > float(cutoff_f)]
                history_rows = work[work["target_id"].astype(str).isin({str(x) for x in history_ids if str(x).strip()})]
                train_rows = pd.concat([history_rows, train_active], ignore_index=False)
                active_cutoff_feasible = (not train_rows.empty) and (not test_active.empty)
        except Exception:
            active_cutoff_feasible = False
    if not active_cutoff_feasible:
        notes.append("active_target_cutoff_not_feasible")

    # Uncertainty feasibility:
    # conformal calibration needs a calibration subset and a "proper" subset.
    unc_cfg = dict(bs.get("uncertainty") or {})
    unc_enabled = bool(unc_cfg.get("enabled", False))
    unc_calib_frac = float(unc_cfg.get("calibration_fraction", 0.2))
    unc_min_calib = _safe_int(unc_cfg.get("min_calibration_rows", 10), 10)

    # Conservative check: need enough rows to allow at least 10 proper-fit rows
    # plus at least min_calibration_rows (or fraction-derived) for calibration.
    # This mirrors `core.response_models._split_for_calibration`.
    unc_feasible = False
    if unc_enabled:
        n = int(len(work))
        calib_size = int((n * unc_calib_frac) // 1)
        calib_size = max(calib_size, unc_min_calib)
        if 0 < calib_size < n and (n - calib_size) >= 10:
            unc_feasible = True
    if not unc_feasible:
        notes.append("uncertainty_calibration_not_feasible")

    # Recommendation backtest feasibility (data-only gate):
    # - needs parameter_constraints to generate candidates
    # - needs at least one split mode to produce folds
    # - needs some recipe columns to exist (the backtest logic is robust but expects these keys)
    param_constraints = config.get("parameter_constraints") or {}
    has_params = bool(param_constraints)
    has_move_cols = all(c in work.columns for c in ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"])
    has_backtest_cols = has_move_cols and ("lifetime" in work.columns) and ("target_id" in work.columns)
    backtest_feasible = bool(has_params and has_backtest_cols and (total_rows >= 20) and (target_count >= 1))
    if not backtest_feasible:
        notes.append("recommendation_backtest_not_feasible")

    # Regime label heuristic.
    min_rows_per_target = min(rows_per_target.values()) if rows_per_target else 0
    if total_rows < 8 or target_count == 0 or min_rows_per_target < 3:
        label = "very_sparse"
    elif target_count == 1:
        label = "single_target_partial"
    elif total_rows >= 60 and target_count >= 3 and min_rows_per_target >= 10:
        label = "rich_multi_target"
    else:
        label = "moderate_multi_target"

    return DataRegime(
        total_rows=total_rows,
        target_count=target_count,
        rows_per_target=rows_per_target,
        leave_one_target_out_feasible=bool(loto_feasible),
        active_target_cutoff_feasible=bool(active_cutoff_feasible),
        uncertainty_calibration_feasible=bool(unc_feasible),
        recommendation_backtest_feasible=bool(backtest_feasible),
        regime_label=label,
        notes=notes,
    )

