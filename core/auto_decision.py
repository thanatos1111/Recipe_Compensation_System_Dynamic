"""
Auto-decision engine backend (separate from manual benchmark workflow).

This module *reuses* existing benchmark / ranking / backtest infrastructure:
- `core.benchmarking.run_prediction_benchmark`
- `core.ranking.rank_benchmark_suite`
- `core.recommendation_backtest.run_recommendation_backtest`

It does not replace or modify the manual benchmark UI flow.
It also does not "adopt" or persist any chosen model bundle; callers must
explicitly apply the result in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from core.benchmarking import BenchmarkSuiteResult, run_prediction_benchmark
from core.bundles import get_effective_model_bundles
from core.data_regime import DataRegime, inspect_data_regime
from core.ranking import rank_benchmark_suite
from core.recommendation_backtest import RecommendationBacktestResult, run_recommendation_backtest


@dataclass(frozen=True)
class AutoDecisionResult:
    detected_regime: DataRegime
    split_modes_used: list[str]
    bundles_evaluated: list[str]
    shortlisted_bundles: list[str]
    winner: Optional[str]
    runner_up: Optional[str]
    confidence_level: str
    explanation_text: str
    excluded_reasons: dict[str, str] = field(default_factory=dict)

    # Optional attached artifacts for downstream UI/inspection.
    benchmark_suite: Optional[BenchmarkSuiteResult] = None
    benchmark_ranked: Optional[list[dict[str, Any]]] = None
    backtest_result: Optional[RecommendationBacktestResult] = None


def build_auto_evaluation_plan(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    Build an auto-evaluation plan (no training).

    Returns a JSON-serializable dict describing:
    - detected regime
    - candidate split modes
    - eligible bundle names
    - excluded reasons
    """
    regime = inspect_data_regime(df, config)
    excluded: dict[str, str] = {}

    split_modes: list[str] = ["forward_chaining"]
    if regime.leave_one_target_out_feasible:
        split_modes.append("leave_one_target_out")
    else:
        excluded["split:leave_one_target_out"] = "not_feasible_in_regime"

    if regime.active_target_cutoff_feasible:
        split_modes.append("active_target_cutoff")
    else:
        excluded["split:active_target_cutoff"] = "not_feasible_or_missing_active_target_config"

    # Bundles: presets + custom, then optional allowlist/denylist.
    effective = get_effective_model_bundles(config)
    bundle_names = sorted(effective.keys())

    ad = config.get("auto_decision") or {}
    allow = ad.get("bundle_allowlist")
    deny = ad.get("bundle_denylist")

    if isinstance(allow, (list, tuple, set)):
        allow_set = {str(x) for x in allow if str(x).strip()}
        before = list(bundle_names)
        bundle_names = [bn for bn in bundle_names if bn in allow_set]
        for bn in before:
            if bn not in bundle_names:
                excluded[f"bundle:{bn}"] = "not_in_allowlist"

    if isinstance(deny, (list, tuple, set)):
        deny_set = {str(x) for x in deny if str(x).strip()}
        kept: list[str] = []
        for bn in bundle_names:
            if bn in deny_set:
                excluded[f"bundle:{bn}"] = "denylisted"
            else:
                kept.append(bn)
        bundle_names = kept

    # If nothing remains, fall back to a small safe default set.
    if not bundle_names:
        # Keep names stable with existing preset keys.
        bundle_names = [bn for bn in ["balanced_default", "baseline_linear", "baseline_tree"] if bn in effective]
        if not bundle_names:
            bundle_names = sorted(effective.keys())[:3]
        excluded["bundles:empty_after_filters"] = "fell_back_to_default_subset"

    return {
        "detected_regime": regime,
        "split_modes": split_modes,
        "eligible_bundles": bundle_names,
        "excluded_reasons": excluded,
    }


def score_auto_decision_candidates(
    *,
    ranked_bundles: list[dict[str, Any]],
    backtest_result: Optional[RecommendationBacktestResult] = None,
    objective: str,
) -> list[dict[str, Any]]:
    """
    Combine benchmark ranking with optional backtest signals for auto-decision.

    Strategy (conservative, deterministic):
    - Use benchmark ranking output as the primary order.
    - If backtest metrics are available for a bundle, attach them for explanation.
    - Optionally re-rank within very tight benchmark ties using backtest improvement rate.
      (We only do this for objectives that are not weighted_combined, to avoid
      inventing new score blends.)
    """
    rows = [dict(r) for r in (ranked_bundles or [])]
    bt_map: dict[str, Any] = {}
    if backtest_result is not None:
        for bn, s in (backtest_result.summaries_by_bundle or {}).items():
            bt_map[str(bn)] = s

    for r in rows:
        bn = str(r.get("bundle_name", ""))
        s = bt_map.get(bn)
        if s is None:
            continue
        r["backtest_rows_evaluated"] = int(getattr(s, "rows_evaluated", 0) or 0)
        r["backtest_recommendation_improvement_rate"] = getattr(s, "recommendation_improvement_rate", None)
        r["backtest_predicted_spec_pass_improvement_rate"] = getattr(s, "predicted_spec_pass_improvement_rate", None)
        r["backtest_no_change_fraction"] = getattr(s, "no_change_fraction", None)

    if objective == "weighted_combined":
        # Avoid tie-breaking heuristics on top of the combined score.
        return rows

    def _tie_group_key(item: dict[str, Any]) -> tuple[Any, Any]:
        # Use the objective's primary/secondary means when present.
        return (item.get("primary_mean"), item.get("secondary_mean"))

    # Stable tie-breaking: within exact (primary, secondary) ties, prefer higher backtest improvement rate.
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(_tie_group_key(r), []).append(r)

    out: list[dict[str, Any]] = []
    for k in [_tie_group_key(r) for r in rows]:
        if k not in grouped:
            continue
        g = grouped.pop(k)
        if len(g) <= 1:
            out.extend(g)
            continue

        def _bt_key(item: dict[str, Any]) -> float:
            v = item.get("backtest_recommendation_improvement_rate")
            try:
                return float(v) if v is not None else float("-inf")
            except Exception:
                return float("-inf")

        # Keep original benchmark order if backtest doesn't distinguish.
        g_sorted = sorted(g, key=lambda d: _bt_key(d), reverse=True)
        out.extend(g_sorted)

    return out


def run_auto_model_selection(df: pd.DataFrame, config: dict[str, Any]) -> AutoDecisionResult:
    """
    Run the full auto-decision flow for one material dataframe.

    Stages
    ------
    1) inspect data regime
    2) choose valid split modes automatically
    3) choose eligible bundles automatically
    4) run prediction benchmark
    5) (optional) run recommendation backtest on shortlisted bundles if feasible
    6) rank and select winner + runner-up
    """
    # Optional cancellation/progress hooks for UI responsiveness.
    abort_event = config.get("_auto_decision_abort_event")
    progress_cb = config.get("_auto_decision_progress_cb")
    progress: Optional[Callable[[int, str], None]] = progress_cb if callable(progress_cb) else None

    def _aborted() -> bool:
        try:
            return bool(abort_event is not None and getattr(abort_event, "is_set", lambda: False)())
        except Exception:
            return False

    def _tick(pct: int, stage: str) -> None:
        if progress is None:
            return
        try:
            progress(int(pct), str(stage))
        except Exception:
            return

    _tick(0, "Planning")
    plan = build_auto_evaluation_plan(df, config)
    regime: DataRegime = plan["detected_regime"]
    split_modes: list[str] = list(plan["split_modes"])
    eligible_bundles: list[str] = list(plan["eligible_bundles"])
    excluded: dict[str, str] = dict(plan.get("excluded_reasons") or {})

    if _aborted():
        excluded["stage:auto_decision"] = "aborted"
        return AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=[],
            bundles_evaluated=[],
            shortlisted_bundles=[],
            winner=None,
            runner_up=None,
            confidence_level="low",
            explanation_text="Auto-decision was aborted before running.",
            excluded_reasons=excluded,
        )

    if df is None or df.empty:
        return AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=[],
            bundles_evaluated=[],
            shortlisted_bundles=[],
            winner=None,
            runner_up=None,
            confidence_level="low",
            explanation_text="No rows available; auto-decision did not run.",
            excluded_reasons=excluded,
            benchmark_suite=None,
            benchmark_ranked=None,
            backtest_result=None,
        )

    effective = get_effective_model_bundles(config)
    model_bundles = {bn: effective[bn] for bn in eligible_bundles if bn in effective}
    bundles_evaluated = sorted(model_bundles.keys())
    if not bundles_evaluated:
        excluded["bundles:none_resolved"] = "no_effective_bundles_available"
        return AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=[],
            bundles_evaluated=[],
            shortlisted_bundles=[],
            winner=None,
            runner_up=None,
            confidence_level="low",
            explanation_text="No eligible bundles resolved from config; auto-decision did not run.",
            excluded_reasons=excluded,
        )

    # Ranking settings (mirror manual benchmark UI defaults when absent).
    bench_cfg = config.get("benchmark_settings") or {}
    objective = str(bench_cfg.get("ranking_objective") or config.get("ranking_objective") or "spec_pass_first").strip()
    uncertainty_mode = str(bench_cfg.get("uncertainty_mode") or config.get("uncertainty_mode") or "ignore").strip().lower()
    if uncertainty_mode not in {"ignore", "warn_only", "include_in_score"}:
        uncertainty_mode = "ignore"
    uncertainty_weights = bench_cfg.get("uncertainty_weights") or config.get("uncertainty_weights")
    weights = bench_cfg.get("ranking_weights") or config.get("ranking_weights")

    conformal_alpha: Optional[float] = None
    unc_cfg = bench_cfg.get("uncertainty") or {}
    if isinstance(unc_cfg, dict) and unc_cfg.get("enabled", False):
        try:
            conformal_alpha = float(unc_cfg.get("alpha", 0.1))
        except Exception:
            conformal_alpha = None

    if _aborted():
        excluded["stage:prediction_benchmark"] = "aborted"
        return AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=split_modes,
            bundles_evaluated=bundles_evaluated,
            shortlisted_bundles=[],
            winner=None,
            runner_up=None,
            confidence_level="low",
            explanation_text="Auto-decision was aborted before benchmarking.",
            excluded_reasons=excluded,
        )

    # Stage 1: prediction benchmark (no fine-grained progress hooks; treat as indeterminate).
    _tick(-1, "Benchmarking (prediction screening)")
    suite = run_prediction_benchmark(
        df,
        config=config,
        model_names_by_target=model_bundles,
        split_modes=split_modes,
    )
    if _aborted():
        excluded["stage:prediction_benchmark"] = "aborted"
        return AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=split_modes,
            bundles_evaluated=bundles_evaluated,
            shortlisted_bundles=[],
            winner=None,
            runner_up=None,
            confidence_level="low",
            explanation_text="Auto-decision was aborted after benchmarking.",
            excluded_reasons=excluded,
            benchmark_suite=suite,
        )

    _tick(-1, "Ranking candidates")
    ranked = rank_benchmark_suite(
        suite,
        objective=objective,
        weights=weights,
        uncertainty_mode=uncertainty_mode,
        uncertainty_weights=uncertainty_weights,
        conformal_alpha=conformal_alpha,
    )

    shortlist_size = 3
    ad = config.get("auto_decision") or {}
    try:
        shortlist_size = int(ad.get("shortlist_size", 3))
    except Exception:
        shortlist_size = 3
    shortlist_size = max(1, min(10, shortlist_size))

    shortlisted = [str(r.get("bundle_name")) for r in ranked[:shortlist_size] if str(r.get("bundle_name", "")).strip()]

    backtest_result: Optional[RecommendationBacktestResult] = None
    if regime.recommendation_backtest_feasible and shortlisted:
        if _aborted():
            excluded["stage:recommendation_backtest"] = "aborted"
            backtest_result = None
        else:
            # Prefer a split mode that exercises multi-target generalization if available.
            preferred_split = "leave_one_target_out" if "leave_one_target_out" in split_modes else split_modes[0]
            try:
                _tick(0, "Recommendation backtest (fold replay)")

                def _bt_progress(done: int, total: int, stage: str) -> None:
                    pct = int((done / total) * 100.0) if total else 0
                    _tick(pct, f"Recommendation backtest: {stage}")

                backtest_result = run_recommendation_backtest(
                    df,
                    config=config,
                    model_bundle_names=shortlisted,
                    split_mode=preferred_split,
                    abort_event=abort_event,
                    progress_cb=_bt_progress if progress is not None else None,
                )
            except Exception:
                excluded["stage:recommendation_backtest"] = "failed_or_not_supported_in_current_config"
                backtest_result = None
    else:
        excluded["stage:recommendation_backtest"] = "not_feasible_in_regime_or_no_shortlist"

    scored = score_auto_decision_candidates(
        ranked_bundles=ranked,
        backtest_result=backtest_result,
        objective=objective,
    )
    winner = str(scored[0]["bundle_name"]) if scored else None
    runner_up = str(scored[1]["bundle_name"]) if len(scored) > 1 else None

    confidence_level = _estimate_confidence(regime, scored=scored, backtest=backtest_result)
    explanation = _build_explanation(
        regime=regime,
        split_modes=split_modes,
        objective=objective,
        uncertainty_mode=uncertainty_mode,
        winner=winner,
        runner_up=runner_up,
        scored=scored,
        backtest=backtest_result,
    )

    return AutoDecisionResult(
        detected_regime=regime,
        split_modes_used=split_modes,
        bundles_evaluated=bundles_evaluated,
        shortlisted_bundles=shortlisted,
        winner=winner,
        runner_up=runner_up,
        confidence_level=confidence_level,
        explanation_text=explanation,
        excluded_reasons=excluded,
        benchmark_suite=suite,
        benchmark_ranked=scored,
        backtest_result=backtest_result,
    )


def _estimate_confidence(
    regime: DataRegime,
    *,
    scored: list[dict[str, Any]],
    backtest: Optional[RecommendationBacktestResult],
) -> str:
    # Conservative heuristic: keep "high" rare.
    if regime.total_rows <= 0 or not scored:
        return "low"

    if regime.regime_label == "very_sparse":
        return "low"

    if regime.regime_label == "single_target_partial":
        return "low" if regime.total_rows < 30 else "moderate"

    # Multi-target regimes
    if regime.regime_label == "rich_multi_target" and regime.leave_one_target_out_feasible:
        if backtest is not None and not backtest.aborted:
            # Require some backtest coverage for the winner.
            winner = str(scored[0].get("bundle_name", ""))
            s = (backtest.summaries_by_bundle or {}).get(winner)
            rows_bt = int(getattr(s, "rows_evaluated", 0) or 0) if s is not None else 0
            if rows_bt >= 10:
                return "high"
        return "moderate"

    return "moderate" if regime.total_rows >= 30 else "low"


def _build_explanation(
    *,
    regime: DataRegime,
    split_modes: list[str],
    objective: str,
    uncertainty_mode: str,
    winner: Optional[str],
    runner_up: Optional[str],
    scored: list[dict[str, Any]],
    backtest: Optional[RecommendationBacktestResult],
) -> str:
    lines: list[str] = []
    lines.append("Auto-decision summary (material-local):")
    lines.append(f"- detected_regime: {regime.regime_label} (rows={regime.total_rows}, targets={regime.target_count})")
    lines.append(f"- split_modes_used: {split_modes}")
    lines.append(f"- ranking_objective: {objective}")
    lines.append(f"- uncertainty_ranking_mode: {uncertainty_mode}")
    if winner:
        lines.append(f"- winner: {winner}")
    if runner_up:
        lines.append(f"- runner_up: {runner_up}")

    if scored:
        top = scored[0]
        p = top.get("primary_mean")
        s = top.get("secondary_mean")
        if p is not None or s is not None:
            lines.append(f"- winner_metrics: primary_mean={p}, secondary_mean={s}")
        bt_rate = top.get("backtest_recommendation_improvement_rate")
        if bt_rate is not None:
            lines.append(f"- winner_backtest_improvement_rate: {bt_rate}")

    if backtest is None:
        lines.append("- recommendation_backtest: skipped")
    else:
        lines.append(f"- recommendation_backtest: ran (aborted={bool(backtest.aborted)})")

    # Keep explanation concise: add key notes.
    if regime.notes:
        lines.append(f"- regime_notes: {sorted(set(regime.notes))}")
    return "\n".join(lines)

