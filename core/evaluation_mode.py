"""
Scenario-based model evaluation for data sufficiency comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.candidate_generator import generate_candidates
from core.feature_engineering import build_feature_matrix
from core.labeling import apply_spec_labels
from core.optimizer import rank_candidates, select_best_candidate
from core.response_models import predict_outputs, train_material_models
from core.schemas import SpecConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    active_target_id: str
    history_target_ids: tuple[str, ...] = ()
    active_cutoff_lifetime: Optional[float] = None


def evaluate_data_sufficiency_scenarios(
    df: pd.DataFrame,
    *,
    spec_config: SpecConfig,
    config: dict[str, Any],
    scenario_a: ScenarioConfig,
    scenario_b: ScenarioConfig,
) -> dict[str, Any]:
    """
    Evaluate two user-defined scenarios on one material dataset.

    The scenarios remain material-local because `df` is already one material.
    """
    a_out = _evaluate_single_scenario(df, spec_config=spec_config, config=config, scenario=scenario_a)
    b_out = _evaluate_single_scenario(df, spec_config=spec_config, config=config, scenario=scenario_b)
    return {
        "scenario_a": a_out,
        "scenario_b": b_out,
        "comparison": _build_comparison(a_out, b_out),
        "interpretation": _build_interpretation(a_out, b_out),
    }


def _evaluate_single_scenario(
    df: pd.DataFrame,
    *,
    spec_config: SpecConfig,
    config: dict[str, Any],
    scenario: ScenarioConfig,
) -> dict[str, Any]:
    train_df, test_df = _build_train_test(df, scenario)
    if train_df.empty:
        return {"scenario": scenario.name, "error": "No training rows in this scenario."}
    if test_df.empty:
        return {"scenario": scenario.name, "error": "No test rows available after cutoff."}

    artifacts = train_material_models(train_df, config=config, spec_config=spec_config)
    model_bundle = {
        "rs_model": artifacts.rs_model,
        "thickness_model": artifacts.thickness_model,
        "rsu_model": artifacts.rsu_model,
        "feature_config": config.get("feature_config", {}),
    }

    X_test = build_feature_matrix(test_df, config.get("feature_config", {}))
    preds = predict_outputs(model_bundle, X_test)
    metrics = _compute_regression_metrics(test_df, preds)
    metrics.update(_compute_spec_quality(test_df, preds, spec_config))
    rec_proxy = _recommendation_proxy(
        test_df,
        model_bundle=model_bundle,
        spec_config=spec_config,
        config=config,
    )

    return {
        "scenario": scenario.name,
        "config": {
            "active_target_id": scenario.active_target_id,
            "history_target_ids": list(scenario.history_target_ids),
            "active_cutoff_lifetime": scenario.active_cutoff_lifetime,
        },
        "row_summary": {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_target_ids": sorted(str(v) for v in train_df["target_id"].astype(str).unique()),
        },
        "metrics": metrics,
        "recommendation_proxy": rec_proxy,
        "confidence_summary": artifacts.confidence_summary,
    }


def _build_train_test(df: pd.DataFrame, scenario: ScenarioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["target_id"] = work["target_id"].astype(str)
    work["lifetime"] = pd.to_numeric(work["lifetime"], errors="coerce")
    work = work.dropna(subset=["target_id", "lifetime"]).sort_values(["target_id", "lifetime"]).reset_index(drop=True)

    history_ids = {str(tid) for tid in scenario.history_target_ids if str(tid).strip()}
    history_df = work[work["target_id"].isin(history_ids)].copy() if history_ids else pd.DataFrame(columns=work.columns)
    active = work[work["target_id"] == scenario.active_target_id].copy()
    cutoff = float(scenario.active_cutoff_lifetime) if scenario.active_cutoff_lifetime is not None else None

    train_parts = [history_df]
    if cutoff is not None:
        train_parts.append(active[active["lifetime"] <= cutoff].copy())
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=work.columns)
    train_df = train_df.sort_values(["target_id", "lifetime"]).drop_duplicates().reset_index(drop=True)

    if cutoff is None:
        test_df = active.copy()
    else:
        test_df = active[active["lifetime"] > cutoff].copy()
    test_df = test_df.sort_values("lifetime").reset_index(drop=True)
    return train_df, test_df


def _compute_regression_metrics(test_df: pd.DataFrame, preds: dict[str, Any]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    pairs = {
        "rs": "rs_pred",
        "thickness": "thickness_pred",
        "rsu": "rsu_pred",
    }
    for actual_col, pred_col in pairs.items():
        y = pd.to_numeric(test_df.get(actual_col), errors="coerce")
        pred_arr = preds.get(pred_col)
        if pred_arr is None:
            out[f"{actual_col}_mae"] = None
            out[f"{actual_col}_rmse"] = None
            continue
        p = pd.Series(np.asarray(pred_arr), index=test_df.index, dtype=float)
        valid = y.notna() & p.notna()
        if int(valid.sum()) == 0:
            out[f"{actual_col}_mae"] = None
            out[f"{actual_col}_rmse"] = None
            continue
        yv = y.loc[valid]
        pv = p.loc[valid]
        out[f"{actual_col}_mae"] = float(mean_absolute_error(yv, pv))
        out[f"{actual_col}_rmse"] = float(np.sqrt(mean_squared_error(yv, pv)))
    return out


def _compute_spec_quality(
    test_df: pd.DataFrame,
    preds: dict[str, Any],
    spec_config: SpecConfig,
) -> dict[str, Optional[float]]:
    if "in_spec" not in test_df.columns:
        return {"spec_pass_accuracy": None}
    pred_table = pd.DataFrame(index=test_df.index)
    for actual_col, pred_col in [("rs", "rs_pred"), ("thickness", "thickness_pred"), ("rsu", "rsu_pred")]:
        arr = preds.get(pred_col)
        if arr is not None:
            pred_table[actual_col] = np.asarray(arr, dtype=float)
    if pred_table.empty:
        return {"spec_pass_accuracy": None}

    labeled = apply_spec_labels(pred_table, spec_config)
    if "in_spec" not in labeled.columns:
        return {"spec_pass_accuracy": None}
    actual = test_df["in_spec"]
    valid = actual.notna() & labeled["in_spec"].notna()
    if int(valid.sum()) == 0:
        return {"spec_pass_accuracy": None}
    acc = (actual.loc[valid].astype(bool) == labeled.loc[valid, "in_spec"].astype(bool)).mean()
    return {"spec_pass_accuracy": float(acc)}


def _recommendation_proxy(
    test_df: pd.DataFrame,
    *,
    model_bundle: dict[str, Any],
    spec_config: SpecConfig,
    config: dict[str, Any],
) -> dict[str, Optional[float]]:
    parameter_config = config.get("parameter_constraints", {}) or {}
    if not parameter_config:
        return {
            "rows_evaluated": 0,
            "baseline_score_mean": None,
            "recommended_score_mean": None,
            "improvement_rate": None,
        }

    weights = config.get("optimizer_weights", {})
    feature_config = config.get("feature_config", {})
    improvements = 0
    baseline_scores: list[float] = []
    rec_scores: list[float] = []
    rows_used = 0

    for _, row in test_df.iterrows():
        reference_recipe = {
            "incident_angle": float(row.get("incident_angle", 0.0) or 0.0),
            "linear_offset": float(row.get("linear_offset", 0.0) or 0.0),
            "rotations": float(row.get("rotations", 0.0) or 0.0),
            "ar_flow": float(row.get("ar_flow", 0.0) or 0.0),
            "o2_flow": float(row.get("o2_flow", 0.0) or 0.0),
            "total_flow": float(row.get("ar_flow", 0.0) or 0.0) + float(row.get("o2_flow", 0.0) or 0.0),
        }
        candidates = generate_candidates(reference_recipe, parameter_config, {"radius_steps": 1, "max_candidates": 120})
        if candidates is None or candidates.empty:
            continue

        baseline = pd.DataFrame([reference_recipe])
        candidates = pd.concat([baseline, candidates], ignore_index=True).drop_duplicates().reset_index(drop=True)
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
        ranked = rank_candidates(candidates, model_bundle, context)
        if ranked.empty or "score" not in ranked.columns:
            continue
        best = select_best_candidate(ranked)
        baseline_rows = ranked[
            (ranked["incident_angle"] == reference_recipe["incident_angle"])
            & (ranked["linear_offset"] == reference_recipe["linear_offset"])
            & (ranked["rotations"] == reference_recipe["rotations"])
            & (ranked["ar_flow"] == reference_recipe["ar_flow"])
            & (ranked["o2_flow"] == reference_recipe["o2_flow"])
        ]
        if baseline_rows.empty:
            continue
        baseline_score = float(baseline_rows.iloc[0]["score"])
        rec_score = float(best.get("score", baseline_score))
        baseline_scores.append(baseline_score)
        rec_scores.append(rec_score)
        if rec_score < baseline_score:
            improvements += 1
        rows_used += 1

    if rows_used == 0:
        return {
            "rows_evaluated": 0,
            "baseline_score_mean": None,
            "recommended_score_mean": None,
            "improvement_rate": None,
        }
    return {
        "rows_evaluated": rows_used,
        "baseline_score_mean": float(np.mean(baseline_scores)),
        "recommended_score_mean": float(np.mean(rec_scores)),
        "improvement_rate": float(improvements / rows_used),
    }


def _build_comparison(a_out: dict[str, Any], b_out: dict[str, Any]) -> dict[str, Optional[float]]:
    if "metrics" not in a_out or "metrics" not in b_out:
        return {}
    deltas: dict[str, Optional[float]] = {}
    for key in [
        "rs_mae",
        "rs_rmse",
        "thickness_mae",
        "thickness_rmse",
        "rsu_mae",
        "rsu_rmse",
        "spec_pass_accuracy",
    ]:
        av = a_out["metrics"].get(key)
        bv = b_out["metrics"].get(key)
        deltas[f"delta_{key}_b_minus_a"] = float(bv - av) if av is not None and bv is not None else None
    return deltas


def _build_interpretation(a_out: dict[str, Any], b_out: dict[str, Any]) -> str:
    if "metrics" not in a_out or "metrics" not in b_out:
        return "Evaluation could not run for one or both scenarios."
    a_rs = a_out["metrics"].get("rs_mae")
    b_rs = b_out["metrics"].get("rs_mae")
    a_rows = a_out.get("row_summary", {}).get("n_train", 0)
    b_rows = b_out.get("row_summary", {}).get("n_train", 0)

    if a_rs is None or b_rs is None:
        return "Insufficient predictions for RS; confidence should be treated as low."
    if b_rows > a_rows and b_rs < a_rs:
        return "Adding richer same-material history improves prediction quality; confidence can be moderately increased."
    if b_rows > a_rows and b_rs >= a_rs:
        return "More data did not improve RS error in this split; keep recommendation confidence conservative."
    return "Scenario differences are small; confidence should stay similar across the compared data windows."
