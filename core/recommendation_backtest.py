"""
Recommendation backtesting (Milestone 6).

This module replays historical rows in a leakage-free, fold-based manner:
- Fit recommendation models only on the fold's train rows.
- For each eligible fold test row:
    - Treat the observed recipe as the baseline.
    - Generate candidate recommendations using only train-fitted models.
    - Compare baseline vs recommendation predicted scores and predicted spec pass.

It produces practical usefulness metrics intended for comparing future models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from core.benchmarking import MODEL_BUNDLE_PRESETS
from core.candidate_generator import generate_candidates
from core.feature_engineering import build_feature_matrix
from core.labeling import apply_spec_labels
from core.optimizer import rank_candidates, select_best_candidate
from core.model_registry import ExternalModelDependencyMissingError
from core.response_models import fit_model_for_target
from core.schemas import SpecConfig
from core.validation_schemes import split_strategy_to_iter


@dataclass(frozen=True)
class RecommendationBacktestSummary:
    bundle_name: str
    rows_evaluated: int
    recommendation_improvement_rate: Optional[float]
    predicted_spec_pass_improvement_rate: Optional[float]
    no_change_fraction: Optional[float]
    move_mean_abs_total: Optional[float]
    move_median_abs_total: Optional[float]
    move_max_abs_total: Optional[float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecommendationBacktestResult:
    split_mode: str
    split_folds: int
    summaries_by_bundle: dict[str, RecommendationBacktestSummary]
    aborted: bool = False

    def to_table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bundle_name, s in self.summaries_by_bundle.items():
            rows.append(
                {
                    "bundle_name": s.bundle_name,
                    "rows_evaluated": s.rows_evaluated,
                    "recommendation_improvement_rate": s.recommendation_improvement_rate,
                    "predicted_spec_pass_improvement_rate": s.predicted_spec_pass_improvement_rate,
                    "no_change_fraction": s.no_change_fraction,
                    "move_mean_abs_total": s.move_mean_abs_total,
                    "move_median_abs_total": s.move_median_abs_total,
                    "move_max_abs_total": s.move_max_abs_total,
                    "warnings": ";".join(s.warnings or []),
                }
            )
        return rows


_MOVE_PARAMS: list[str] = ["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"]


def _reference_recipe_from_row(row: pd.Series) -> dict[str, Any]:
    """Build the recipe parameter dict used by candidate generation."""
    ar_flow = float(row.get("ar_flow", 0.0) or 0.0)
    o2_flow = float(row.get("o2_flow", 0.0) or 0.0)
    total_flow = ar_flow + o2_flow
    return {
        "incident_angle": float(row.get("incident_angle", 0.0) or 0.0),
        "linear_offset": float(row.get("linear_offset", 0.0) or 0.0),
        "rotations": float(row.get("rotations", 0.0) or 0.0),
        "ar_flow": ar_flow,
        "o2_flow": o2_flow,
        "total_flow": total_flow,
    }


def _predicted_in_spec_for_ranked_row(ranked_row: pd.Series, spec_config: SpecConfig) -> Optional[bool]:
    """
    Compute spec-pass label from a ranked-candidate row.

    The ranker includes predicted outputs as ``*_pred`` columns.
    """
    # Build a minimal "predicted outputs" frame compatible with apply_spec_labels.
    pred: dict[str, Any] = {}
    if getattr(spec_config, "use_rs_spec", True) and "rs_pred" in ranked_row:
        pred["rs"] = float(ranked_row.get("rs_pred"))
    if getattr(spec_config, "use_thickness_spec", True) and "thickness_pred" in ranked_row:
        pred["thickness"] = float(ranked_row.get("thickness_pred"))
    if getattr(spec_config, "use_rsu_spec", True) and "rsu_pred" in ranked_row:
        pred["rsu"] = float(ranked_row.get("rsu_pred"))

    if not pred:
        return None

    pred_df = pd.DataFrame([pred])
    labeled = apply_spec_labels(pred_df, spec_config)
    if "in_spec" not in labeled.columns:
        return None
    v = labeled["in_spec"].iloc[0]
    if pd.isna(v):
        return None
    return bool(v)


def score_recommendation_outcome(
    actual_row: pd.Series,
    recommended_candidate: dict[str, Any],
    predicted_outputs: dict[str, Any],
    spec_config: SpecConfig,
) -> dict[str, Any]:
    """
    Score a recommendation outcome for one test row.

    This provides a row-level, practical usefulness summary that can be aggregated.
    """
    # Actual spec result for the baseline recipe.
    actual_in_spec: Optional[bool]
    if "in_spec" in actual_row.index:
        val = actual_row.get("in_spec")
        actual_in_spec = None if pd.isna(val) else bool(val)
    else:
        actual_pred: dict[str, Any] = {}
        if getattr(spec_config, "use_rs_spec", True) and "rs" in actual_row.index:
            actual_pred["rs"] = float(actual_row.get("rs"))
        if getattr(spec_config, "use_thickness_spec", True) and "thickness" in actual_row.index:
            actual_pred["thickness"] = float(actual_row.get("thickness"))
        if getattr(spec_config, "use_rsu_spec", True) and "rsu" in actual_row.index:
            actual_pred["rsu"] = float(actual_row.get("rsu"))
        actual_in_spec = None
        if actual_pred:
            labeled = apply_spec_labels(pd.DataFrame([actual_pred]), spec_config)
            actual_in_spec = bool(labeled["in_spec"].iloc[0])

    # Predicted spec result for the recommended candidate.
    pred_in_spec: Optional[bool]
    pred_out: dict[str, Any] = {}

    def _to_scalar(x: Any) -> Optional[float]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return float(x[0])
        if hasattr(x, "shape") and getattr(x, "shape", None) is not None:
            try:
                arr = np.asarray(x, dtype=float)
                if arr.shape == ():
                    return float(arr)
                if arr.size == 1:
                    return float(arr.reshape(-1)[0])
            except Exception:
                return None
        try:
            return float(x)
        except Exception:
            return None

    rs_pred = _to_scalar(predicted_outputs.get("rs_pred"))
    th_pred = _to_scalar(predicted_outputs.get("thickness_pred"))
    rsu_pred = _to_scalar(predicted_outputs.get("rsu_pred"))
    if rs_pred is not None:
        pred_out["rs"] = rs_pred
    if th_pred is not None:
        pred_out["thickness"] = th_pred
    if rsu_pred is not None:
        pred_out["rsu"] = rsu_pred
    pred_in_spec = None
    if pred_out:
        labeled_pred = apply_spec_labels(pd.DataFrame([pred_out]), spec_config)
        pred_in_spec = bool(labeled_pred["in_spec"].iloc[0])

    return {
        "actual_in_spec": actual_in_spec,
        "recommended_pred_in_spec": pred_in_spec,
        "predicted_spec_pass_improved": bool(pred_in_spec) and (actual_in_spec is False),
        "no_change_or_conservative": (actual_in_spec is not None and pred_in_spec is not None and bool(actual_in_spec) == bool(pred_in_spec)),
    }


def backtest_single_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: dict[str, Any],
    model_bundle_name: str,
    max_test_rows_per_fold: Optional[int] = None,
    abort_event: Optional[Any] = None,
    progress_tick: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    """
    Backtest one fold for one model bundle preset.

    Returns dict with counts + move vectors so callers can aggregate across folds.
    """
    if train_df is None or train_df.empty or test_df is None or test_df.empty:
        return {
            "aborted": False,
            "rows_evaluated": 0,
            "improvement_rows": 0,
            "predicted_spec_improved_rows": 0,
            "no_change_rows": 0,
            "move_abs_totals": [],
            "warnings": [],
        }

    parameter_config = config.get("parameter_constraints") or {}
    if not parameter_config:
        return {
            "aborted": False,
            "rows_evaluated": 0,
            "improvement_rows": 0,
            "predicted_spec_improved_rows": 0,
            "no_change_rows": 0,
            "move_abs_totals": [],
            "warnings": [],
        }

    spec_config = config.get("spec_config")
    if spec_config is None or not isinstance(spec_config, SpecConfig):
        # Spec metrics become impossible without a SpecConfig.
        # Still allow scoring-based backtest (improvement rate on predicted score).
        spec_config = SpecConfig()

    weights = config.get("optimizer_weights") or {}
    feature_config = config.get("feature_config") or {}

    preset = MODEL_BUNDLE_PRESETS.get(model_bundle_name) or {}
    rs_model_name = preset.get("rs", config.get("model_settings", {}).get("rs_model", "gbr"))
    thickness_model_name = preset.get("thickness", config.get("model_settings", {}).get("thickness_model", "gbr"))
    rsu_model_name = preset.get("rsu", config.get("model_settings", {}).get("rsu_model", "gbr"))

    # Fit regression pipelines on fold train rows only.
    X_train = build_feature_matrix(train_df, feature_config)
    model_cfg = dict(config.get("model_settings") or {})
    model_cfg["rs_model"] = rs_model_name
    model_cfg["thickness_model"] = thickness_model_name
    model_cfg["rsu_model"] = rsu_model_name
    fit_cfg = dict(config)
    fit_cfg["model_settings"] = model_cfg

    fold_warnings: list[str] = []

    try:
        rs_model = fit_model_for_target(train_df, "rs", fit_cfg, feature_matrix=X_train)
    except ExternalModelDependencyMissingError as exc:
        fold_warnings.append(exc.user_message)
        rs_model = None

    try:
        th_model = fit_model_for_target(train_df, "thickness", fit_cfg, feature_matrix=X_train)
    except ExternalModelDependencyMissingError as exc:
        fold_warnings.append(exc.user_message)
        th_model = None

    try:
        rsu_model = fit_model_for_target(train_df, "rsu", fit_cfg, feature_matrix=X_train)
    except ExternalModelDependencyMissingError as exc:
        fold_warnings.append(exc.user_message)
        rsu_model = None

    model_bundle = {
        "rs_model": rs_model,
        "thickness_model": th_model,
        "rsu_model": rsu_model,
        "feature_config": feature_config,
    }

    improvements = 0
    predicted_spec_improvements = 0
    no_change_rows = 0
    move_abs_totals: list[float] = []
    rows_evaluated = 0

    # Neighborhood for replay: fixed small radius to match the recommendation proxy.
    neighborhood_config = {"radius_steps": 1, "max_candidates": 120}

    # Precompute eligible columns for performance.
    required_cols = set(_MOVE_PARAMS + ["lifetime", "rpm", "power", "target_id"])
    _ = required_cols  # placeholder to satisfy potential future extensions

    backtest_cfg = config.get("backtest_settings") or {}
    # UI/backtest can otherwise become very expensive on large datasets.
    # Default to a modest cap; callers can override (0/None => no cap).
    max_test_rows = max_test_rows_per_fold if max_test_rows_per_fold is not None else backtest_cfg.get("max_test_rows_per_fold", 50)
    try:
        max_test_rows_int = int(max_test_rows) if max_test_rows is not None else 0
    except Exception:
        max_test_rows_int = 50

    if max_test_rows_int > 0 and len(test_df) > max_test_rows_int:
        test_df = test_df.head(max_test_rows_int)

    def _aborted() -> bool:
        return bool(abort_event is not None and getattr(abort_event, "is_set", lambda: False)())

    if _aborted():
        return {
            "aborted": True,
            "rows_evaluated": 0,
            "improvement_rows": 0,
            "predicted_spec_improved_rows": 0,
            "no_change_rows": 0,
            "move_abs_totals": [],
            "warnings": fold_warnings,
        }

    for _, row in test_df.iterrows():
        if _aborted():
            return {
                "aborted": True,
                "rows_evaluated": rows_evaluated,
                "improvement_rows": improvements,
                "predicted_spec_improved_rows": predicted_spec_improvements,
                "no_change_rows": no_change_rows,
                "move_abs_totals": move_abs_totals,
                "warnings": fold_warnings,
            }

        reference_recipe = _reference_recipe_from_row(row)

        candidates = generate_candidates(reference_recipe, parameter_config, neighborhood_config)
        if candidates is None or candidates.empty:
            if progress_tick is not None:
                progress_tick()
            continue

        # Ensure baseline recipe is always present in ranking.
        baseline = pd.DataFrame([reference_recipe])
        candidates_all = pd.concat([baseline, candidates], ignore_index=True).drop_duplicates().reset_index(drop=True)

        context = {
            "spec_config": spec_config,
            "weights": weights,
            "reference_recipe": reference_recipe,
            "active_target_id": str(row.get("target_id", "")),
            "lifetime": float(row.get("lifetime", 0.0) or 0.0),
            "rpm": float(row["rpm"]) if pd.notna(row.get("rpm")) else None,
            "power": float(row["power"]) if pd.notna(row.get("power")) else None,
            "feature_config": feature_config,
            "instance_correction": None,
            "parameter_config": parameter_config,
        }

        ranked = rank_candidates(candidates_all, model_bundle, context)
        if ranked is None or ranked.empty or "score" not in ranked.columns:
            if progress_tick is not None:
                progress_tick()
            continue

        best = select_best_candidate(ranked)
        if not best:
            if progress_tick is not None:
                progress_tick()
            continue

        # Locate baseline row inside ranked by exact equality on varied parameters.
        baseline_mask = (
            (ranked.get("incident_angle") == reference_recipe["incident_angle"])
            & (ranked.get("linear_offset") == reference_recipe["linear_offset"])
            & (ranked.get("rotations") == reference_recipe["rotations"])
            & (ranked.get("ar_flow") == reference_recipe["ar_flow"])
            & (ranked.get("o2_flow") == reference_recipe["o2_flow"])
        )
        try:
            baseline_rows = ranked.loc[baseline_mask]
        except Exception:
            baseline_rows = pd.DataFrame()

        if baseline_rows.empty:
            if progress_tick is not None:
                progress_tick()
            continue

        baseline_score = float(baseline_rows.iloc[0]["score"])
        rec_score = float(best.get("score", baseline_score))

        rows_evaluated += 1
        if rec_score < baseline_score:
            improvements += 1

        # Spec outcomes: actual baseline spec result vs predicted recommendation spec result.
        recommended_predicted_outputs = {
            "rs_pred": best.get("rs_pred"),
            "thickness_pred": best.get("thickness_pred"),
            "rsu_pred": best.get("rsu_pred"),
        }
        spec_outcome = score_recommendation_outcome(
            row,
            recommended_candidate=best,
            predicted_outputs=recommended_predicted_outputs,
            spec_config=spec_config,
        )

        if spec_outcome["predicted_spec_pass_improved"]:
            predicted_spec_improvements += 1
        if spec_outcome["no_change_or_conservative"]:
            no_change_rows += 1

        # Move size (absolute) between baseline and recommendation.
        move_total = 0.0
        for p in _MOVE_PARAMS:
            base_v = float(reference_recipe.get(p, 0.0) or 0.0)
            rec_v = best.get(p)
            rec_v_f = float(rec_v) if rec_v is not None and not pd.isna(rec_v) else base_v
            move_total += abs(rec_v_f - base_v)
        move_abs_totals.append(float(move_total))

        if progress_tick is not None:
            progress_tick()

    if rows_evaluated == 0:
        return {
            "aborted": False,
            "rows_evaluated": 0,
            "improvement_rows": 0,
            "predicted_spec_improved_rows": 0,
            "no_change_rows": 0,
            "move_abs_totals": [],
            "warnings": fold_warnings,
        }

    return {
        "aborted": False,
        "rows_evaluated": rows_evaluated,
        "improvement_rows": improvements,
        "predicted_spec_improved_rows": predicted_spec_improvements,
        "no_change_rows": no_change_rows,
        "move_abs_totals": move_abs_totals,
        "warnings": fold_warnings,
    }


def run_recommendation_backtest(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    model_bundle_names: Any,
    split_mode: str,
    max_test_rows_per_fold: Optional[int] = None,
    abort_event: Optional[Any] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> RecommendationBacktestResult:
    """
    Run recommendation backtests across split folds and model bundles.
    """
    if df is None or df.empty:
        return RecommendationBacktestResult(
            split_mode=split_mode,
            split_folds=0,
            summaries_by_bundle={},
            aborted=False,
        )

    bs = config.get("benchmark_settings") or {}
    st_key = (split_mode or "").strip().lower()
    if st_key in ("leave_one_target_out", "loto"):
        extra = dict(bs.get("leave_one_target_out", {}) or {})
    elif st_key in ("active_target_cutoff", "active_cutoff"):
        extra = dict(bs.get("active_target_cutoff", {}) or {})
    else:
        extra = dict(bs.get("forward_chaining", {}) or {})

    # Normalize model bundle names: accept presets list or dict mappings.
    bundle_names: list[str] = []
    if model_bundle_names is None:
        bundle_names = []
    elif isinstance(model_bundle_names, (list, tuple)):
        bundle_names = [str(x) for x in model_bundle_names]
    elif isinstance(model_bundle_names, dict):
        # Treat as either {bundle_name: {...}} or a single mapping.
        if all(k in model_bundle_names for k in ("rs", "thickness", "rsu")):
            bundle_names = ["custom"]
        else:
            bundle_names = [str(k) for k in model_bundle_names.keys()]
    else:
        bundle_names = [str(model_bundle_names)]

    if not bundle_names:
        return RecommendationBacktestResult(split_mode=split_mode, split_folds=0, summaries_by_bundle={})

    spec_config = config.get("spec_config")
    if spec_config is None:
        # UI should always pass spec_config, but keep module resilient.
        spec_config = SpecConfig()

    # Ensure backtest uses SpecConfig consistently.
    config_for_backtest = dict(config)
    config_for_backtest["spec_config"] = spec_config

    backtest_cfg = config.get("backtest_settings") or {}
    max_cap_raw = max_test_rows_per_fold if max_test_rows_per_fold is not None else backtest_cfg.get("max_test_rows_per_fold", 50)
    try:
        max_cap_int = int(max_cap_raw) if max_cap_raw is not None else 0
    except Exception:
        max_cap_int = 50

    split_iter = list(split_strategy_to_iter(df, split_mode, extra))
    fold_count = 0
    # Aggregate per bundle.
    acc: dict[str, dict[str, Any]] = {}
    for bn in bundle_names:
        acc[bn] = {
            "rows_evaluated": 0,
            "improvement_rows": 0,
            "predicted_spec_improved_rows": 0,
            "no_change_rows": 0,
            "move_abs_totals": [],
            "warnings": [],
        }

    # Compute total progress units: capped test rows across all folds times number of bundles.
    fold_test_row_caps: list[int] = []
    for _split_name, _split_type, _train_idx, test_idx in split_iter:
        n_test = int(len(df.loc[test_idx]))
        cap = n_test if max_cap_int <= 0 else min(n_test, max_cap_int)
        fold_test_row_caps.append(cap)
    total_units = int(sum(fold_test_row_caps)) * max(len(bundle_names), 1)

    done_units = 0
    last_pct = -1
    aborted_any = False

    def _progress_tick(stage: str) -> None:
        nonlocal done_units, last_pct
        if progress_cb is None:
            return
        done_units += 1
        if total_units <= 0:
            pct = 0
        else:
            pct = int((done_units / total_units) * 100)
        if pct != last_pct:
            last_pct = pct
            progress_cb(done_units, total_units, stage)

    for fold_i, (_split_name, _split_type, train_idx, test_idx) in enumerate(split_iter):
        fold_count += 1
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        for bn in bundle_names:
            stage = f"fold {fold_i + 1}/{len(split_iter)} - {bn}"
            fold_metrics = backtest_single_fold(
                train_df,
                test_df,
                config=config_for_backtest,
                model_bundle_name=bn,
                max_test_rows_per_fold=max_cap_int,
                abort_event=abort_event,
                progress_tick=lambda st=stage: _progress_tick(st),
            )
            if int(fold_metrics.get("rows_evaluated") or 0) == 0:
                continue
            acc_bn = acc[bn]
            acc_bn["rows_evaluated"] += int(fold_metrics["rows_evaluated"])
            acc_bn["improvement_rows"] += int(fold_metrics["improvement_rows"])
            acc_bn["predicted_spec_improved_rows"] += int(fold_metrics["predicted_spec_improved_rows"])
            acc_bn["no_change_rows"] += int(fold_metrics["no_change_rows"])
            acc_bn["move_abs_totals"].extend([float(v) for v in fold_metrics.get("move_abs_totals") or []])
            acc_bn["warnings"].extend([str(w) for w in fold_metrics.get("warnings") or []])

            if bool(fold_metrics.get("aborted")):
                aborted_any = True
                break

        if aborted_any:
            break

    summaries: dict[str, RecommendationBacktestSummary] = {}
    for bn in bundle_names:
        a = acc[bn]
        rows_evaluated = int(a["rows_evaluated"])
        if rows_evaluated == 0:
            summaries[bn] = RecommendationBacktestSummary(
                bundle_name=bn,
                rows_evaluated=0,
                recommendation_improvement_rate=None,
                predicted_spec_pass_improvement_rate=None,
                no_change_fraction=None,
                move_mean_abs_total=None,
                move_median_abs_total=None,
                move_max_abs_total=None,
                warnings=acc[bn].get("warnings") or [],
            )
            continue

        improvement_rate = float(a["improvement_rows"]) / float(rows_evaluated)
        predicted_spec_rate = float(a["predicted_spec_improved_rows"]) / float(rows_evaluated)
        no_change_fraction = float(a["no_change_rows"]) / float(rows_evaluated)

        moves = np.asarray(a["move_abs_totals"], dtype=float)
        move_mean = float(np.mean(moves)) if moves.size else None
        move_median = float(np.median(moves)) if moves.size else None
        move_max = float(np.max(moves)) if moves.size else None

        summaries[bn] = RecommendationBacktestSummary(
            bundle_name=bn,
            rows_evaluated=rows_evaluated,
            recommendation_improvement_rate=improvement_rate,
            predicted_spec_pass_improvement_rate=predicted_spec_rate,
            no_change_fraction=no_change_fraction,
            move_mean_abs_total=move_mean,
            move_median_abs_total=move_median,
            move_max_abs_total=move_max,
            warnings=acc[bn].get("warnings") or [],
        )

    return RecommendationBacktestResult(
        split_mode=split_mode,
        split_folds=fold_count,
        summaries_by_bundle=summaries,
        aborted=aborted_any,
    )

