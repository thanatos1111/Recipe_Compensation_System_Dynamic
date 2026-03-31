"""
Objective-driven ranking for benchmark results.

This module separates "how to rank" (objective) from benchmark execution so the
same logic can be reused by the manual benchmark UI and future auto-decision
features.
"""

from __future__ import annotations

from typing import Any, Optional

_ERROR_METRICS: frozenset[str] = frozenset({"rs_mae", "thickness_mae", "rsu_mae"})
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"spec_pass_accuracy", "interval_quality"})

_UNCERTAINTY_MODES: frozenset[str] = frozenset({"ignore", "warn_only", "include_in_score"})

# Per-response keys on BenchmarkSummary / dict summaries.
_INTERVAL_AGG_SPEC: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("rs", ("rs_interval_coverage_mean", "rs_interval_mean_width_mean", "rs_interval_median_width_mean")),
    ("thickness", ("thickness_interval_coverage_mean", "thickness_interval_mean_width_mean", "thickness_interval_median_width_mean")),
    ("rsu", ("rsu_interval_coverage_mean", "rsu_interval_mean_width_mean", "rsu_interval_median_width_mean")),
)


def get_supported_uncertainty_modes() -> dict[str, dict[str, Any]]:
    """Return UI-friendly metadata for uncertainty handling during ranking."""

    return {
        "ignore": {
            "display_name": "Ignore uncertainty",
            "description": "Rank using accuracy / error metrics only.",
        },
        "warn_only": {
            "display_name": "Warn only",
            "description": "Same ranking as ignore, but attach coverage warnings when intervals look miscalibrated.",
        },
        "include_in_score": {
            "display_name": "Include in score",
            "description": "Penalize poor interval calibration and overly wide intervals (relative to other bundles).",
        },
    }


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
    uncertainty_weights: Optional[dict[str, float]] = None,
    conformal_alpha: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Rank model bundles by aggregating summary metrics across all split modes.

    Parameters
    - results: typically a `core.benchmarking.BenchmarkSuiteResult`, but any object
      with a `.runs` iterable containing `.bundle_name` and `.summary` works.
    - objective: one of `get_supported_ranking_objectives().keys()`.
    - weights: only used for `weighted_combined`.
    - uncertainty_mode: ``ignore``, ``warn_only``, or ``include_in_score``.
    - uncertainty_weights: optional tuning for interval quality (coverage vs width,
      blend into weighted combined). See `_default_uncertainty_weights()`.
    - conformal_alpha: when set, nominal target coverage is ``1 - alpha``; otherwise
      a neutral default (0.9) is used for coverage closeness and warnings.
    """

    _validate_objective(objective)
    _validate_uncertainty_mode(uncertainty_mode)

    by_bundle: dict[str, list[Any]] = {}
    for run in getattr(results, "runs", []) or []:
        bundle_name = str(getattr(run, "bundle_name", ""))
        by_bundle.setdefault(bundle_name, []).append(getattr(run, "summary", None))

    uw = _merge_uncertainty_weights(uncertainty_weights)
    scored: list[dict[str, Any]] = []
    for bundle_name, summaries in by_bundle.items():
        scored_item = score_benchmark_bundle(
            summaries,
            objective=objective,
            weights=weights,
            uncertainty_mode=uncertainty_mode,
            uncertainty_weights=uw,
            conformal_alpha=conformal_alpha,
        )
        scored_item["bundle_name"] = bundle_name
        scored.append(scored_item)

    if uncertainty_mode == "include_in_score":
        _apply_uncertainty_quality_across_bundles(scored, uncertainty_weights=uw)

    if objective == "weighted_combined":
        scored = _attach_weighted_normalized_scores(
            scored,
            weights=weights,
            uncertainty_mode=uncertainty_mode,
            uncertainty_weights=uw,
        )
        return sorted(scored, key=_weighted_sort_key)

    return sorted(scored, key=lambda d: _lexicographic_sort_key(d, objective=objective, uncertainty_mode=uncertainty_mode))


def score_benchmark_bundle(
    run_summaries: list[Any],
    *,
    objective: str,
    weights: Optional[dict[str, float]] = None,
    uncertainty_mode: str = "ignore",
    uncertainty_weights: Optional[dict[str, float]] = None,
    conformal_alpha: Optional[float] = None,
) -> dict[str, Any]:
    """
    Aggregate split-mode summaries for one bundle into objective-relevant fields.

    Returns a dict that always contains the aggregated means for:
    - spec_pass_accuracy
    - rs_mae
    - thickness_mae
    - rsu_mae
    plus objective-specific convenience fields.

    When uncertainty is enabled, ``interval_aggregates`` holds per-target coverage
    and width means. ``uncertainty_warnings`` is populated in ``warn_only`` mode.
    ``uncertainty_quality`` is filled in ``include_in_score`` only after
    ``rank_benchmark_suite`` computes cross-bundle width context; until then it is
    absent (single-bundle callers get a neutral default from
    `_apply_uncertainty_quality_across_bundles`).
    """

    _validate_objective(objective)
    _validate_uncertainty_mode(uncertainty_mode)
    uw = _merge_uncertainty_weights(uncertainty_weights)

    spec = _mean_ignore_none([_get_metric_value(s, "spec_pass_accuracy") for s in run_summaries])
    rs = _mean_ignore_none([_get_metric_value(s, "rs_mae") for s in run_summaries])
    th = _mean_ignore_none([_get_metric_value(s, "thickness_mae") for s in run_summaries])
    rsu = _mean_ignore_none([_get_metric_value(s, "rsu_mae") for s in run_summaries])

    target_cov = _target_interval_coverage(conformal_alpha)
    interval_agg = _aggregate_interval_metrics(run_summaries)

    out: dict[str, Any] = {
        "spec_pass_accuracy_mean": spec,
        "rs_mae_mean": rs,
        "thickness_mae_mean": th,
        "rsu_mae_mean": rsu,
        "interval_aggregates": interval_agg,
        "target_interval_coverage": float(target_cov),
    }

    if uncertainty_mode == "warn_only":
        out["uncertainty_warnings"] = _coverage_warnings_for_aggregates(interval_agg, target_cov)

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
        out["effective_uncertainty_weights"] = dict(uw)
    else:
        raise ValueError(f"Unsupported objective: {objective!r}")

    if objective != "weighted_combined":
        out["effective_uncertainty_weights"] = dict(uw)

    return out


# -----------------------
# Internal helpers
# -----------------------


def _default_uncertainty_weights() -> dict[str, float]:
    return {
        "interval_quality": 1.0,
        "coverage": 1.0,
        "width": 1.0,
        "median_width": 0.0,
    }


def _merge_uncertainty_weights(uncertainty_weights: Optional[dict[str, float]]) -> dict[str, float]:
    out = _default_uncertainty_weights()
    if not uncertainty_weights:
        return out
    for k, v in uncertainty_weights.items():
        key = str(k)
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


def _validate_objective(objective: str) -> None:
    supported = get_supported_ranking_objectives()
    if objective not in supported:
        raise ValueError(f"Unknown objective: {objective!r}. Supported: {sorted(supported.keys())}")


def _validate_uncertainty_mode(mode: str) -> None:
    if mode not in _UNCERTAINTY_MODES:
        raise ValueError(
            f"Unknown uncertainty_mode: {mode!r}. Supported: {sorted(_UNCERTAINTY_MODES)}",
        )


def _target_interval_coverage(conformal_alpha: Optional[float]) -> float:
    """
    Nominal target interval coverage.

    When conformal alpha is provided, use ``1 - alpha``. Otherwise use a neutral
    default compatible with common ``alpha=0.1`` benchmarks.
    """

    if conformal_alpha is None:
        return 0.9
    try:
        a = float(conformal_alpha)
    except Exception:
        return 0.9
    if not (0.0 < a < 1.0):
        return 0.9
    return float(max(0.0, min(1.0, 1.0 - a)))


def _mean_ignore_none(xs: list[Optional[float]]) -> Optional[float]:
    xs2 = [float(v) for v in xs if v is not None]
    if not xs2:
        return None
    return float(sum(xs2) / len(xs2))


def _aggregate_interval_metrics(run_summaries: list[Any]) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {}
    for resp, (kc, km, kde) in _INTERVAL_AGG_SPEC:
        covs = [_get_summary_scalar(s, kc) for s in run_summaries]
        means = [_get_summary_scalar(s, km) for s in run_summaries]
        meds = [_get_summary_scalar(s, kde) for s in run_summaries]
        out[resp] = {
            "coverage": _mean_ignore_none(covs),
            "mean_width": _mean_ignore_none(means),
            "median_width": _mean_ignore_none(meds),
        }
    return out


def _get_summary_scalar(summary: Any, key: str) -> Optional[float]:
    if summary is None:
        return None
    if isinstance(summary, dict):
        return _coerce_optional_float(summary.get(key))
    return _coerce_optional_float(getattr(summary, key, None))


def _coverage_warnings_for_aggregates(
    interval_agg: dict[str, dict[str, Optional[float]]],
    target: float,
    *,
    tolerance: float = 0.05,
) -> list[str]:
    labels = {"rs": "RS", "thickness": "Thickness", "rsu": "RSU"}
    warnings: list[str] = []
    for resp, label in labels.items():
        row = interval_agg.get(resp) or {}
        cov = row.get("coverage")
        if cov is None:
            continue
        try:
            c = float(cov)
        except Exception:
            continue
        delta = c - float(target)
        if abs(delta) <= float(tolerance):
            continue
        if delta < 0:
            warnings.append(
                f"{label} interval coverage {c:.3f} is below target {float(target):.3f} "
                "(possible under-coverage / miscalibration).",
            )
        else:
            warnings.append(
                f"{label} interval coverage {c:.3f} exceeds target {float(target):.3f} "
                "(intervals may be conservative).",
            )
    return warnings


def _width_scores_population(values: list[Optional[float]]) -> dict[int, float]:
    """Map index -> normalized score in [0,1], lower width = higher score. Missing -> 0.5."""

    resolved: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        fv = _coerce_optional_float(v)
        if fv is None:
            continue
        resolved.append((i, float(fv)))
    if not resolved:
        return {i: 0.5 for i in range(len(values))}
    nums = [fv for _, fv in resolved]
    wmin, wmax = min(nums), max(nums)
    out: dict[int, float] = {i: 0.5 for i in range(len(values))}
    for i, fv in resolved:
        if wmax == wmin:
            out[i] = 0.5
        else:
            out[i] = float((wmax - fv) / (wmax - wmin))
    return out


def _coverage_closeness(cov: float, target: float) -> float:
    denom = max(float(target), 1.0 - float(target), 1e-6)
    return float(max(0.0, min(1.0, 1.0 - abs(float(cov) - float(target)) / denom)))


def _response_uncertainty_score(
    row: dict[str, Optional[float]],
    target: float,
    mean_score: float,
    median_score: float,
    uw: dict[str, float],
) -> Optional[float]:
    w_cov = max(0.0, float(uw.get("coverage", 1.0)))
    w_w = max(0.0, float(uw.get("width", 1.0)))
    mb = max(0.0, min(1.0, float(uw.get("median_width", 0.0))))

    cov = row.get("coverage")
    mean_w = row.get("mean_width")
    median_w = row.get("median_width")

    parts: list[float] = []
    weights: list[float] = []

    if cov is not None:
        try:
            parts.append(_coverage_closeness(float(cov), target))
            weights.append(w_cov)
        except Exception:
            pass

    if mean_w is not None or median_w is not None:
        width_part = (1.0 - mb) * mean_score + mb * median_score
        parts.append(float(width_part))
        weights.append(w_w)

    if not parts or sum(weights) <= 0.0:
        return None
    return float(sum(p * w for p, w in zip(parts, weights)) / sum(weights))


def _apply_uncertainty_quality_across_bundles(scored: list[dict[str, Any]], *, uncertainty_weights: dict[str, float]) -> None:
    """Set ``uncertainty_quality`` on each item (neutral 0.5 when insufficient data)."""

    n = len(scored)
    if n == 0:
        return

    target = float(scored[0].get("target_interval_coverage", 0.9))

    for resp, _fields in _INTERVAL_AGG_SPEC:
        mean_vals = []
        median_vals = []
        for item in scored:
            agg = item.get("interval_aggregates") or {}
            row = agg.get(resp) or {}
            mean_vals.append(row.get("mean_width"))
            median_vals.append(row.get("median_width"))
        mean_norm = _width_scores_population(mean_vals)
        med_norm = _width_scores_population(median_vals)
        for idx, item in enumerate(scored):
            agg = item.get("interval_aggregates") or {}
            row = dict(agg.get(resp) or {})
            mscore = mean_norm.get(idx, 0.5)
            mdscore = med_norm.get(idx, 0.5)
            q = _response_uncertainty_score(row, target, mscore, mdscore, uncertainty_weights)
            if "_interval_quality_parts" not in item:
                item["_interval_quality_parts"] = {}
            item["_interval_quality_parts"][str(resp)] = q

    for item in scored:
        parts_map = item.get("_interval_quality_parts") or {}
        vals = [v for v in parts_map.values() if v is not None]
        if vals:
            item["uncertainty_quality"] = float(sum(vals) / len(vals))
        else:
            item["uncertainty_quality"] = 0.5
        item.pop("_interval_quality_parts", None)


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


def _lexicographic_sort_key(
    item: dict[str, Any],
    *,
    objective: str,
    uncertainty_mode: str,
) -> tuple[Any, ...]:
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
        base: tuple[Any, ...] = (-p_val, s_val)
    elif objective in {"rs_first", "thickness_first", "rsu_first"}:
        # primary: MAE (lower), secondary: spec_pass_accuracy (higher)
        p_val = float(p) if p is not None else float("inf")
        s_val = float(s) if s is not None else float("-inf")
        base = (p_val, -s_val)
    else:
        raise ValueError(f"Unsupported objective for lexicographic sort: {objective!r}")

    if uncertainty_mode == "include_in_score":
        uq = item.get("uncertainty_quality")
        u_val = float(uq) if uq is not None else 0.5
        return (*base, -u_val, bundle_name)
    return (*base, bundle_name)


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
    uncertainty_mode: str = "ignore",
    uncertainty_weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    w = dict(_default_weights())
    if weights:
        for k, v in weights.items():
            try:
                w[str(k)] = float(v)
            except Exception:
                # Ignore non-numeric weights gracefully.
                continue

    uw = _merge_uncertainty_weights(uncertainty_weights)
    iqw = max(0.0, float(uw.get("interval_quality", 1.0)))

    metrics = ["spec_pass_accuracy", "rs_mae", "thickness_mae", "rsu_mae"]
    if uncertainty_mode == "include_in_score" and iqw > 0.0:
        metrics = [*metrics, "interval_quality"]

    # Precompute per-metric min/max over available (non-None) values.
    mm: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for m in metrics:
        vals: list[float] = []
        for item in scored:
            if m == "interval_quality":
                v = item.get("uncertainty_quality")
            else:
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
            if m == "interval_quality":
                weight = iqw
            else:
                weight = float(w.get(m, 0.0))
            if weight == 0.0:
                continue

            if m == "interval_quality":
                v = item.get("uncertainty_quality")
            else:
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
        if uncertainty_mode == "include_in_score" and iqw > 0.0:
            new_item["effective_weights"]["interval_quality"] = float(iqw)
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
