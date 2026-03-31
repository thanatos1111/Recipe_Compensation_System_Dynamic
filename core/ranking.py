"""
Objective-driven ranking for benchmark results.

This module separates "how to rank" (objective) from benchmark execution so the
same logic can be reused by the manual benchmark UI and future auto-decision
features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


_ERROR_METRICS: frozenset[str] = frozenset({"rs_mae", "thickness_mae", "rsu_mae"})
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"spec_pass_accuracy"})


def get_supported_ranking_objectives() -> dict[str, dict[str, Any]]:
    """
    Return UI-friendly metadata for supported ranking objectives.

    Keys are stable objective identifiers; values are metadata intended for UI.
    """

    return {
        "spec_pass_first": {
            "display_name": "Spec-pass first",
            "description": "Prioritize higher spec-pass accuracy; tie-break with lower RS MAE.",
            "uses_custom_weights": False,
        },
        "rs_first": {
            "display_name": "RS first",
            "description": "Prioritize lower RS MAE; tie-break with higher spec-pass accuracy.",
            "uses_custom_weights": False,
        },
        "thickness_first": {
            "display_name": "Thickness first",
            "description": "Prioritize lower Thickness MAE; tie-break with higher spec-pass accuracy.",
            "uses_custom_weights": False,
        },
        "rsu_first": {
            "display_name": "RSU first",
            "description": "Prioritize lower RSU MAE; tie-break with higher spec-pass accuracy.",
            "uses_custom_weights": False,
        },
        "weighted_combined": {
            "display_name": "Weighted combined",
            "description": "Compute a normalized weighted score across summary metrics (direction-aware).",
            "uses_custom_weights": True,
        },
    }


def rank_benchmark_suite(
    results: Any,
    *,
    objective: str,
    weights: Optional[dict[str, float]] = None,
    uncertainty_mode: str = "ignore",
) -> list[dict[str, Any]]:
    """
    Rank model bundles by aggregating summary metrics across all split modes.

    Parameters
    - results: typically a `core.benchmarking.BenchmarkSuiteResult`, but any object
      with a `.runs` iterable containing `.bundle_name` and `.summary` works.
    - objective: one of `get_supported_ranking_objectives().keys()`.
    - weights: only used for `weighted_combined`.
    - uncertainty_mode: reserved for future uncertainty-aware ranking; currently
      only "ignore" is supported.
    """

    _validate_objective(objective)
    if uncertainty_mode != "ignore":
        raise ValueError("Only uncertainty_mode='ignore' is currently supported.")

    by_bundle: dict[str, list[Any]] = {}
    for run in getattr(results, "runs", []) or []:
        bundle_name = str(getattr(run, "bundle_name", ""))
        by_bundle.setdefault(bundle_name, []).append(getattr(run, "summary", None))

    scored: list[dict[str, Any]] = []
    for bundle_name, summaries in by_bundle.items():
        scored_item = score_benchmark_bundle(
            summaries,
            objective=objective,
            weights=weights,
            uncertainty_mode=uncertainty_mode,
        )
        scored_item["bundle_name"] = bundle_name
        scored.append(scored_item)

    if objective == "weighted_combined":
        scored = _attach_weighted_normalized_scores(scored, weights=weights)
        return sorted(scored, key=_weighted_sort_key)

    return sorted(scored, key=lambda d: _lexicographic_sort_key(d, objective=objective))


def score_benchmark_bundle(
    run_summaries: list[Any],
    *,
    objective: str,
    weights: Optional[dict[str, float]] = None,
    uncertainty_mode: str = "ignore",
) -> dict[str, Any]:
    """
    Aggregate split-mode summaries for one bundle into objective-relevant fields.

    Returns a dict that always contains the aggregated means for:
    - spec_pass_accuracy
    - rs_mae
    - thickness_mae
    - rsu_mae
    plus objective-specific convenience fields.
    """

    _validate_objective(objective)
    if uncertainty_mode != "ignore":
        raise ValueError("Only uncertainty_mode='ignore' is currently supported.")

    spec = _mean_ignore_none([_get_metric_value(s, "spec_pass_accuracy") for s in run_summaries])
    rs = _mean_ignore_none([_get_metric_value(s, "rs_mae") for s in run_summaries])
    th = _mean_ignore_none([_get_metric_value(s, "thickness_mae") for s in run_summaries])
    rsu = _mean_ignore_none([_get_metric_value(s, "rsu_mae") for s in run_summaries])

    out: dict[str, Any] = {
        "spec_pass_accuracy_mean": spec,
        "rs_mae_mean": rs,
        "thickness_mae_mean": th,
        "rsu_mae_mean": rsu,
    }

    if objective == "spec_pass_first":
        out["primary_mean"] = spec
        out["secondary_mean"] = rs
    elif objective == "rs_first":
        out["primary_mean"] = rs
        out["secondary_mean"] = spec
    elif objective == "thickness_first":
        out["primary_mean"] = th
        out["secondary_mean"] = spec
    elif objective == "rsu_first":
        out["primary_mean"] = rsu
        out["secondary_mean"] = spec
    elif objective == "weighted_combined":
        # Combined score is attached in `rank_benchmark_suite` because it needs
        # population-wide normalization across bundles.
        out["weights"] = dict(weights or {})
    else:
        raise ValueError(f"Unsupported objective: {objective!r}")

    return out


# -----------------------
# Internal helpers
# -----------------------


def _validate_objective(objective: str) -> None:
    supported = get_supported_ranking_objectives()
    if objective not in supported:
        raise ValueError(f"Unknown objective: {objective!r}. Supported: {sorted(supported.keys())}")


def _mean_ignore_none(xs: list[Optional[float]]) -> Optional[float]:
    xs2 = [float(v) for v in xs if v is not None]
    if not xs2:
        return None
    return float(sum(xs2) / len(xs2))


def _get_metric_value(summary: Any, metric: str) -> Optional[float]:
    """
    Best-effort metric accessor.

    Supports:
    - `core.schemas.BenchmarkSummary` (attributes like `rs_mae_mean`)
    - dict-like summaries with keys `rs_mae_mean` or `rs_mae`
    """

    # Allow dict summaries in tests / lightweight use.
    if isinstance(summary, dict):
        if metric == "spec_pass_accuracy":
            return _coerce_optional_float(summary.get("spec_pass_accuracy_mean", summary.get("spec_pass_accuracy")))
        if metric in _ERROR_METRICS:
            return _coerce_optional_float(summary.get(f"{metric}_mean", summary.get(metric)))
        return _coerce_optional_float(summary.get(f"{metric}_mean", summary.get(metric)))

    if metric == "spec_pass_accuracy":
        return _coerce_optional_float(getattr(summary, "spec_pass_accuracy_mean", None))
    if metric == "rs_mae":
        return _coerce_optional_float(getattr(summary, "rs_mae_mean", None))
    if metric == "thickness_mae":
        return _coerce_optional_float(getattr(summary, "thickness_mae_mean", None))
    if metric == "rsu_mae":
        return _coerce_optional_float(getattr(summary, "rsu_mae_mean", None))

    # Future-friendly direct access (e.g., "rs_rmse") if present.
    return _coerce_optional_float(getattr(summary, f"{metric}_mean", None))


def _coerce_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _lexicographic_sort_key(item: dict[str, Any], *, objective: str) -> tuple[float, float, str]:
    """
    Deterministic sorting for the non-weighted objectives.

    For each objective we define:
    - primary direction (higher or lower)
    - secondary direction (higher or lower)
    Missing primary/secondary values are treated as worst possible.
    """

    bundle_name = str(item.get("bundle_name", ""))

    p = item.get("primary_mean")
    s = item.get("secondary_mean")

    if objective == "spec_pass_first":
        # primary: spec_pass_accuracy (higher), secondary: rs_mae (lower)
        p_val = float(p) if p is not None else float("-inf")
        s_val = float(s) if s is not None else float("inf")
        return (-p_val, s_val, bundle_name)

    if objective in {"rs_first", "thickness_first", "rsu_first"}:
        # primary: MAE (lower), secondary: spec_pass_accuracy (higher)
        p_val = float(p) if p is not None else float("inf")
        s_val = float(s) if s is not None else float("-inf")
        return (p_val, -s_val, bundle_name)

    raise ValueError(f"Unsupported objective for lexicographic sort: {objective!r}")


def _default_weights() -> dict[str, float]:
    return {
        "spec_pass_accuracy": 1.0,
        "rs_mae": 1.0,
        "thickness_mae": 1.0,
        "rsu_mae": 1.0,
    }


def _attach_weighted_normalized_scores(
    scored: list[dict[str, Any]],
    *,
    weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    w = dict(_default_weights())
    if weights:
        for k, v in weights.items():
            try:
                w[str(k)] = float(v)
            except Exception:
                # Ignore non-numeric weights gracefully.
                continue

    metrics = ["spec_pass_accuracy", "rs_mae", "thickness_mae", "rsu_mae"]
    # Precompute per-metric min/max over available (non-None) values.
    mm: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for m in metrics:
        vals: list[float] = []
        for item in scored:
            v = item.get(f"{m}_mean")
            if v is None:
                continue
            vals.append(float(v))
        if not vals:
            mm[m] = (None, None)
        else:
            mm[m] = (min(vals), max(vals))

    out: list[dict[str, Any]] = []
    for item in scored:
        norm_parts: dict[str, float] = {}
        total = 0.0
        total_w = 0.0

        for m in metrics:
            weight = float(w.get(m, 0.0))
            if weight == 0.0:
                continue

            v = item.get(f"{m}_mean")
            mmin, mmax = mm[m]
            norm = _normalize_metric(m, v, mmin=mmin, mmax=mmax)
            norm_parts[m] = float(norm)
            total += weight * float(norm)
            total_w += weight

        combined = (total / total_w) if total_w > 0.0 else 0.0
        new_item = dict(item)
        new_item["normalized_components"] = norm_parts
        new_item["weighted_score"] = float(combined)
        new_item["effective_weights"] = {k: float(v) for k, v in w.items()}
        out.append(new_item)

    return out


def _normalize_metric(metric: str, v: Any, *, mmin: Optional[float], mmax: Optional[float]) -> float:
    """
    Map a metric value to [0, 1] where 1 means "best".

    Missing values yield 0.0 (worst) to avoid missing-metric bundles ranking
    above well-measured bundles.
    """

    fv = _coerce_optional_float(v)
    if fv is None:
        return 0.0
    if mmin is None or mmax is None:
        return 0.0
    if mmax == mmin:
        return 0.5

    if metric in _HIGHER_IS_BETTER:
        return float((fv - mmin) / (mmax - mmin))
    if metric in _ERROR_METRICS:
        return float((mmax - fv) / (mmax - mmin))

    # Unknown metrics: assume lower is better to be conservative.
    return float((mmax - fv) / (mmax - mmin))


def _weighted_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    # Higher weighted_score is better; missing treated as worst.
    s = item.get("weighted_score")
    s_val = float(s) if s is not None else float("-inf")
    return (-s_val, str(item.get("bundle_name", "")))

