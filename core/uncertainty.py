"""
Model-agnostic uncertainty utilities.

This milestone implements simple conformal prediction intervals based on
absolute residuals from a held-out calibration split.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def fit_residual_conformal_calibrator(
    y_true: Any, y_pred: Any, alpha: float = 0.1
) -> dict[str, Any]:
    """
    Fit a residual conformal calibrator using absolute residuals.

    The produced intervals are two-sided:
        [y_pred - q, y_pred + q]
    where q is a (conformal) quantile of the calibration residuals.
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    residuals = np.abs(y_true_arr[mask] - y_pred_arr[mask])
    n = int(residuals.shape[0])

    if n == 0:
        return {
            "alpha": alpha,
            "valid": False,
            "calibration_size": 0,
            "residual_quantile": None,
        }

    # Conformal quantile (order statistic) for finite-sample coverage.
    # Let k = ceil((n + 1) * (1 - alpha)). Use the k-th smallest residual
    # (1-indexed) which corresponds to index (k - 1) in 0-indexed arrays.
    level = 1.0 - alpha
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)

    # partition is O(n) and does not fully sort the array.
    q = float(np.partition(residuals, k - 1)[k - 1])

    return {
        "alpha": alpha,
        "valid": True,
        "calibration_size": n,
        "residual_quantile": q,
        "residual_abs_mean": float(np.mean(residuals)) if n > 0 else None,
        "residual_abs_median": float(np.median(residuals)) if n > 0 else None,
    }


def apply_conformal_interval(y_pred: Any, calibrator: dict[str, Any]) -> pd.DataFrame:
    """
    Apply a fitted residual conformal calibrator to point predictions.

    Returns a DataFrame with columns:
      - ``lower``
      - ``upper``
    preserving the index of the input if it is a pandas Series.
    """
    q = calibrator.get("residual_quantile")
    if q is None or (isinstance(q, float) and np.isnan(q)):
        q = None

    if isinstance(y_pred, pd.Series):
        idx = y_pred.index
        pred_arr = y_pred.to_numpy(dtype=float)
    else:
        idx = None
        pred_arr = np.asarray(y_pred, dtype=float)

    if q is None:
        lower = np.full(pred_arr.shape, np.nan, dtype=float)
        upper = np.full(pred_arr.shape, np.nan, dtype=float)
    else:
        lower = pred_arr - float(q)
        upper = pred_arr + float(q)

    out = pd.DataFrame({"lower": lower, "upper": upper})
    if idx is not None:
        out.index = idx
    return out


def evaluate_interval_quality(
    y_true: Any, lower: Any, upper: Any
) -> dict[str, Optional[float]]:
    """
    Evaluate interval quality.

    Metrics:
      - coverage: fraction of points with lower <= y_true <= upper
      - mean interval width
      - median interval width
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    if y_true_arr.shape != lower_arr.shape or y_true_arr.shape != upper_arr.shape:
        raise ValueError("y_true, lower, upper must have the same shape.")

    # Enforce correct ordering even if upstream data is malformed.
    lo = np.minimum(lower_arr, upper_arr)
    hi = np.maximum(lower_arr, upper_arr)

    valid = ~np.isnan(y_true_arr) & ~np.isnan(lo) & ~np.isnan(hi)
    if int(valid.sum()) == 0:
        return {"coverage": None, "mean_interval_width": None, "median_interval_width": None}

    yv = y_true_arr[valid]
    lov = lo[valid]
    hiv = hi[valid]

    coverage = float(((yv >= lov) & (yv <= hiv)).mean())
    widths = (hiv - lov).astype(float)

    return {
        "coverage": coverage,
        "mean_interval_width": float(np.mean(widths)) if widths.size else None,
        "median_interval_width": float(np.median(widths)) if widths.size else None,
    }

