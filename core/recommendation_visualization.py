"""
Helpers to build recommendation-result visualization payloads.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _series_to_record(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in row.index:
        val = row[col]
        out[str(col)] = None if pd.isna(val) else val
    return out


def build_recommendation_visualization_payload(
    ranked_df: pd.DataFrame,
    *,
    top_n: int = 10,
    selected_index: int = 0,
    parameter_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a compact, UI-friendly payload from ranked candidates.

    The payload is visualization-only and does not modify ranking behavior.
    """
    if ranked_df is None or ranked_df.empty:
        return {
            "best": {},
            "selected": {},
            "score_breakdown": {},
            "top_n_table": [],
            "parameter_prediction_deltas": [],
            "selected_index": 0,
            "top_n_count": 0,
        }

    top_n = max(1, int(top_n))
    show = ranked_df.head(top_n).reset_index(drop=True)
    idx = int(max(0, min(int(selected_index), len(show) - 1)))

    best_row = show.iloc[0]
    selected_row = show.iloc[idx]

    if parameter_order is None:
        parameter_order = [c for c in ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"] if c in show.columns]
    else:
        parameter_order = [p for p in parameter_order if p in show.columns]

    score_cols = [c for c in ["score", "rs_error_norm", "thickness_error_norm", "rsu_penalty", "move_penalty", "safe_margin_reward"] if c in show.columns]
    pred_cols = [c for c in ["rs_pred", "thickness_pred", "rsu_pred"] if c in show.columns]

    score_breakdown = {c: _as_float(best_row.get(c)) for c in score_cols}
    top_table_cols = [c for c in ["score", *parameter_order, *pred_cols] if c in show.columns]
    top_n_table = show[top_table_cols].to_dict(orient="records")

    parameter_prediction_deltas: list[dict[str, Any]] = []
    for p in parameter_order:
        item: dict[str, Any] = {
            "parameter": p,
            "best_value": _as_float(best_row.get(p)),
            "selected_value": _as_float(selected_row.get(p)),
            "delta_parameter": None,
            "delta_rs_pred": None,
            "delta_thickness_pred": None,
            "delta_rsu_pred": None,
        }
        b = _as_float(best_row.get(p))
        s = _as_float(selected_row.get(p))
        if b is not None and s is not None:
            item["delta_parameter"] = s - b
        for pred in ["rs_pred", "thickness_pred", "rsu_pred"]:
            bv = _as_float(best_row.get(pred))
            sv = _as_float(selected_row.get(pred))
            if bv is not None and sv is not None:
                item[f"delta_{pred}"] = sv - bv
        parameter_prediction_deltas.append(item)

    return {
        "best": _series_to_record(best_row),
        "selected": _series_to_record(selected_row),
        "score_breakdown": score_breakdown,
        "top_n_table": top_n_table,
        "parameter_prediction_deltas": parameter_prediction_deltas,
        "selected_index": idx,
        "top_n_count": len(show),
    }

