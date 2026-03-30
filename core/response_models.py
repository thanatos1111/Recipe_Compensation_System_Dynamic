"""
Material-specific response model training + prediction (placeholder).

Milestone 4 will implement model training and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from core.feature_engineering import build_feature_matrix, get_feature_schema
from core.labeling import apply_spec_labels
from core.schemas import BenchmarkFoldResult, BenchmarkSummary, MaterialModelArtifacts, SpecConfig
from core.model_registry import build_preprocessed_regression_pipeline
from core.validation_schemes import split_strategy_to_iter


def _model_name_for_target(target_col: str, config: dict[str, Any]) -> str:
    """
    Resolve model registry name for a target regression output.

    Per-target keys supported:
    - ``rs_model``
    - ``thickness_model``
    - ``rsu_model``
    """
    ms = config.get("model_settings", {}) or {}
    fallback = str(ms.get("rs_model", "gbr"))
    return str(ms.get(f"{target_col}_model", fallback))


def build_model_pipeline(target_col: str, config: dict[str, Any]) -> Pipeline:
    """
    Construct an unfitted sklearn ``Pipeline`` (preprocess + regressor) for ``target_col``.

    Model choice mirrors ``_train_regression_model`` / prior milestone behavior.
    """
    feature_schema = get_feature_schema(config.get("feature_config", {}))
    ms = config.get("model_settings", {}) or {}
    random_state = int(ms.get("random_state", 0))
    model_name = _model_name_for_target(target_col, config)
    return build_preprocessed_regression_pipeline(
        model_name,
        numeric_features=list(feature_schema.numeric_features),
        categorical_features=list(feature_schema.categorical_features),
        config=config,
        random_state=random_state,
    )


def fit_model_for_target(
    df_train: pd.DataFrame,
    target_col: str,
    config: dict[str, Any],
    *,
    feature_matrix: Optional[pd.DataFrame] = None,
) -> Any:
    """Fit a regression pipeline on ``df_train`` for ``target_col``."""
    if df_train is None or df_train.empty:
        return None
    feature_config = config.get("feature_config", {})
    X = feature_matrix if feature_matrix is not None else build_feature_matrix(df_train, feature_config)
    if X is None or X.empty:
        return None

    feature_schema = get_feature_schema(feature_config)
    numeric_features = [c for c in feature_schema.numeric_features if c in X.columns]
    categorical_features = [c for c in feature_schema.categorical_features if c in X.columns]

    y = pd.to_numeric(df_train[target_col], errors="coerce")
    valid = y.notna()
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]

    if len(y_valid) < 10:
        return None

    pipeline = build_model_pipeline(target_col, config)
    # Restrict ColumnTransformer to columns present in X (same as legacy path).
    prep = pipeline.named_steps["preprocess"]
    prep.transformers = [
        ("num", prep.transformers[0][1], [c for c in numeric_features if c in X_valid.columns]),
        ("cat", prep.transformers[1][1], [c for c in categorical_features if c in X_valid.columns]),
    ]
    pipeline.fit(X_valid, y_valid)
    return pipeline


def predict_target(model: Any, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    """Run regression predictions; returns ``None`` if the model is missing or empty."""
    if model is None or X_test is None or X_test.empty:
        return None
    return np.asarray(model.predict(X_test))


def train_rs_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RS regression model."""
    return fit_model_for_target(df, "rs", config, feature_matrix=feature_matrix)


def train_thickness_model(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Train a thickness regression model."""
    return fit_model_for_target(df, "thickness", config, feature_matrix=feature_matrix)


def train_rsu_model(df: pd.DataFrame, feature_matrix: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Train an RSU regression model."""
    return fit_model_for_target(df, "rsu", config, feature_matrix=feature_matrix)


def train_spec_classifier(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Optional classifier for in-spec probability."""
    del df, feature_matrix, config
    return None


def predict_outputs(model_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Predict RS, thickness, and RSU (and optional in-spec probability)."""
    rs_model = model_bundle.get("rs_model")
    thickness_model = model_bundle.get("thickness_model")
    rsu_model = model_bundle.get("rsu_model")

    out: dict[str, Any] = {}
    if rs_model is not None:
        out["rs_pred"] = rs_model.predict(X)
    if thickness_model is not None:
        out["thickness_pred"] = thickness_model.predict(X)
    if rsu_model is not None:
        out["rsu_pred"] = rsu_model.predict(X)
    return out


def _regression_mae_rmse(
    y: pd.Series,
    pred: Optional[np.ndarray],
) -> tuple[Optional[float], Optional[float]]:
    if pred is None:
        return None, None
    p = pd.Series(np.asarray(pred, dtype=float), index=y.index)
    valid = y.notna() & p.notna()
    if int(valid.sum()) == 0:
        return None, None
    yv = y.loc[valid].astype(float)
    pv = p.loc[valid].astype(float)
    mae = float(mean_absolute_error(yv, pv))
    rmse = float(np.sqrt(mean_squared_error(yv, pv)))
    return mae, rmse


def _spec_pass_accuracy(
    test_df: pd.DataFrame,
    preds: dict[str, Any],
    spec_config: SpecConfig,
) -> Optional[float]:
    if "in_spec" not in test_df.columns:
        return None
    pred_table = pd.DataFrame(index=test_df.index)
    for actual_col, pred_col in [("rs", "rs_pred"), ("thickness", "thickness_pred"), ("rsu", "rsu_pred")]:
        arr = preds.get(pred_col)
        if arr is not None:
            pred_table[actual_col] = np.asarray(arr, dtype=float)
    if pred_table.empty:
        return None
    labeled = apply_spec_labels(pred_table, spec_config)
    if "in_spec" not in labeled.columns:
        return None
    actual = test_df["in_spec"]
    valid = actual.notna() & labeled["in_spec"].notna()
    if int(valid.sum()) == 0:
        return None
    acc = (actual.loc[valid].astype(bool) == labeled.loc[valid, "in_spec"].astype(bool)).mean()
    return float(acc)


def _aggregate_folds(
    folds: list[BenchmarkFoldResult],
    split_type: str,
    *,
    model_names: dict[str, str],
) -> BenchmarkSummary:
    def mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
        xs = [float(v) for v in values if v is not None]
        if not xs:
            return None, None
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    if not folds:
        return BenchmarkSummary(split_type=split_type, folds=[], model_names=model_names)

    keys = [
        "rs_mae",
        "rs_rmse",
        "thickness_mae",
        "thickness_rmse",
        "rsu_mae",
        "rsu_rmse",
        "spec_pass_accuracy",
    ]
    agg: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for k in keys:
        vals = [getattr(f, k) for f in folds]
        agg[k] = mean_std(vals)

    return BenchmarkSummary(
        split_type=split_type,
        folds=folds,
        model_names=model_names,
        rs_mae_mean=agg["rs_mae"][0],
        rs_mae_std=agg["rs_mae"][1],
        rs_rmse_mean=agg["rs_rmse"][0],
        rs_rmse_std=agg["rs_rmse"][1],
        thickness_mae_mean=agg["thickness_mae"][0],
        thickness_mae_std=agg["thickness_mae"][1],
        thickness_rmse_mean=agg["thickness_rmse"][0],
        thickness_rmse_std=agg["thickness_rmse"][1],
        rsu_mae_mean=agg["rsu_mae"][0],
        rsu_mae_std=agg["rsu_mae"][1],
        rsu_rmse_mean=agg["rsu_rmse"][0],
        rsu_rmse_std=agg["rsu_rmse"][1],
        spec_pass_accuracy_mean=agg["spec_pass_accuracy"][0],
        spec_pass_accuracy_std=agg["spec_pass_accuracy"][1],
    )


def evaluate_models_with_splits(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    split_strategy: str,
    spec_config: Optional[SpecConfig] = None,
) -> dict[str, Any]:
    """
    Leakage-free benchmark: for each split, fit only on train rows and score on test rows.

    Returns serializable dict with ``folds``, ``summary`` (as dicts), and flat aggregate keys.
    """
    if df is None or df.empty:
        return {
            "folds": [],
            "summary": asdict(BenchmarkSummary(split_type=split_strategy, folds=[])),
            "benchmark_split_type": split_strategy,
            "n_folds": 0,
            "rs_mae": None,
            "thickness_mae": None,
            "rsu_mae": None,
        }

    bs = config.get("benchmark_settings") or {}
    st_key = (split_strategy or "").strip().lower()
    if st_key in ("leave_one_target_out", "loto"):
        extra = dict(bs.get("leave_one_target_out", {}) or {})
    elif st_key in ("active_target_cutoff", "active_cutoff"):
        extra = dict(bs.get("active_target_cutoff", {}) or {})
    else:
        extra = dict(bs.get("forward_chaining", {}) or {})

    fold_results: list[BenchmarkFoldResult] = []
    warnings_global: list[str] = []

    try:
        split_iter = split_strategy_to_iter(df, split_strategy, extra)
    except ValueError as exc:
        warnings_global.append(str(exc))
        split_iter = iter(())

    feature_config = config.get("feature_config", {})
    model_names = {
        "rs": _model_name_for_target("rs", config),
        "thickness": _model_name_for_target("thickness", config),
        "rsu": _model_name_for_target("rsu", config),
    }

    for split_name, split_type, train_idx, test_idx in split_iter:
        tr = train_idx.intersection(df.index)
        te = test_idx.intersection(df.index)
        if len(tr) == 0 or len(te) == 0 or tr.intersection(te).any():
            fold_results.append(
                BenchmarkFoldResult(
                    split_name=split_name,
                    split_type=split_type,
                    train_row_count=0,
                    test_row_count=0,
                    train_target_ids=[],
                    test_target_ids=[],
                    rs_model_name=model_names["rs"],
                    thickness_model_name=model_names["thickness"],
                    rsu_model_name=model_names["rsu"],
                    warnings=["invalid_split"],
                )
            )
            continue

        df_train = df.loc[tr]
        df_test = df.loc[te]
        X_test = build_feature_matrix(df_test, feature_config)

        rs_model = fit_model_for_target(df_train, "rs", config)
        th_model = fit_model_for_target(df_train, "thickness", config)
        rsu_model = fit_model_for_target(df_train, "rsu", config)

        preds: dict[str, Any] = {}
        pr_rs = predict_target(rs_model, X_test)
        pr_th = predict_target(th_model, X_test)
        pr_rsu = predict_target(rsu_model, X_test)
        if pr_rs is not None:
            preds["rs_pred"] = pr_rs
        if pr_th is not None:
            preds["thickness_pred"] = pr_th
        if pr_rsu is not None:
            preds["rsu_pred"] = pr_rsu

        y_rs = pd.to_numeric(df_test["rs"], errors="coerce") if "rs" in df_test.columns else pd.Series(dtype=float)
        y_th = pd.to_numeric(df_test["thickness"], errors="coerce") if "thickness" in df_test.columns else pd.Series(
            dtype=float
        )
        y_rsu = pd.to_numeric(df_test["rsu"], errors="coerce") if "rsu" in df_test.columns else pd.Series(dtype=float)

        rs_mae, rs_rmse = _regression_mae_rmse(y_rs, pr_rs)
        th_mae, th_rmse = _regression_mae_rmse(y_th, pr_th)
        rsu_mae, rsu_rmse = _regression_mae_rmse(y_rsu, pr_rsu)

        spec_acc: Optional[float] = None
        if spec_config is not None:
            spec_acc = _spec_pass_accuracy(df_test, preds, spec_config)

        tid_col = "target_id"
        train_tids = sorted({str(x) for x in df_train[tid_col].astype(str).unique()}) if tid_col in df_train.columns else []
        test_tids = sorted({str(x) for x in df_test[tid_col].astype(str).unique()}) if tid_col in df_test.columns else []

        fold_results.append(
            BenchmarkFoldResult(
                split_name=split_name,
                split_type=split_type,
                train_row_count=int(len(df_train)),
                test_row_count=int(len(df_test)),
                train_target_ids=train_tids,
                test_target_ids=test_tids,
                rs_mae=rs_mae,
                rs_rmse=rs_rmse,
                thickness_mae=th_mae,
                thickness_rmse=th_rmse,
                rsu_mae=rsu_mae,
                rsu_rmse=rsu_rmse,
                spec_pass_accuracy=spec_acc,
                rs_model_name=model_names["rs"],
                thickness_model_name=model_names["thickness"],
                rsu_model_name=model_names["rsu"],
                warnings=[],
            )
        )

    # Determine split_type from first fold or strategy
    stype = fold_results[0].split_type if fold_results else split_strategy
    summary = _aggregate_folds(fold_results, stype, model_names=model_names)
    if warnings_global:
        summary.aggregate_warnings.extend(warnings_global)

    summary_dict = asdict(summary)
    flat: dict[str, Any] = {
        "benchmark_split_type": split_strategy,
        "n_folds": len(fold_results),
        "folds": [asdict(f) for f in fold_results],
        "summary": summary_dict,
        "rs_mae": summary.rs_mae_mean,
        "rs_mae_std": summary.rs_mae_std,
        "rs_rmse": summary.rs_rmse_mean,
        "rs_rmse_std": summary.rs_rmse_std,
        "thickness_mae": summary.thickness_mae_mean,
        "thickness_mae_std": summary.thickness_mae_std,
        "thickness_rmse": summary.thickness_rmse_mean,
        "thickness_rmse_std": summary.thickness_rmse_std,
        "rsu_mae": summary.rsu_mae_mean,
        "rsu_mae_std": summary.rsu_mae_std,
        "rsu_rmse": summary.rsu_rmse_mean,
        "rsu_rmse_std": summary.rsu_rmse_std,
        "spec_pass_accuracy": summary.spec_pass_accuracy_mean,
        "spec_pass_accuracy_std": summary.spec_pass_accuracy_std,
        "n_train": float(np.mean([f.train_row_count for f in fold_results])) if fold_results else None,
        "n_test": float(np.mean([f.test_row_count for f in fold_results])) if fold_results else None,
    }
    if summary.aggregate_warnings:
        flat["benchmark_warnings"] = summary.aggregate_warnings
    return flat


def train_material_models(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    spec_config: Optional[SpecConfig] = None,
) -> MaterialModelArtifacts:
    """
    Train per-material models for RS, thickness, and RSU on all rows (deployment).

    Reported metrics come from leakage-free split-based evaluation, not from scoring
    the deployment fit on a holdout that was seen during training.
    """
    feature_config = config.get("feature_config", {})
    schema = get_feature_schema(feature_config)
    X = build_feature_matrix(df, feature_config)

    model_names = {
        "rs": _model_name_for_target("rs", config),
        "thickness": _model_name_for_target("thickness", config),
        "rsu": _model_name_for_target("rsu", config),
    }

    rs_model = train_rs_model(df, X, config)
    thickness_model = train_thickness_model(df, X, config)
    rsu_model = train_rsu_model(df, X, config)

    bs = config.get("benchmark_settings") or {}
    split_strategy = str(bs.get("split_strategy", "forward_chaining"))

    metrics = evaluate_models_with_splits(
        df,
        config=config,
        split_strategy=split_strategy,
        spec_config=spec_config,
    )

    train_summary = {
        "row_count": int(len(df)),
        "in_spec_count": int(df["in_spec"].sum()) if "in_spec" in df.columns else None,
        "target_instance_count": int(df["target_id"].nunique()) if "target_id" in df.columns else None,
    }

    confidence_summary = {
        "level": _confidence_level(df),
        "notes": _confidence_notes(df),
    }

    return MaterialModelArtifacts(
        rs_model=rs_model,
        thickness_model=thickness_model,
        rsu_model=rsu_model,
        rs_model_name=model_names["rs"],
        thickness_model_name=model_names["thickness"],
        rsu_model_name=model_names["rsu"],
        model_names=model_names,
        spec_classifier=None,
        feature_schema={"numeric": schema.numeric_features, "categorical": schema.categorical_features},
        metrics=metrics,
        train_summary=train_summary,
        confidence_summary=confidence_summary,
    )


def _confidence_level(df: pd.DataFrame) -> str:
    n = len(df) if df is not None else 0
    if n < 10:
        return "Low"
    if n < 25:
        return "Medium"
    return "High"


def _confidence_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if df is None or df.empty:
        return ["no_data"]
    if len(df) < 10:
        notes.append("very_few_rows")
    if "target_id" in df.columns and df["target_id"].nunique() <= 1:
        notes.append("single_target_instance_only")
    if "lifetime" in df.columns:
        lifetime = pd.to_numeric(df["lifetime"], errors="coerce")
        if lifetime.notna().sum() > 0:
            span = float(lifetime.max() - lifetime.min())
            if span <= 0:
                notes.append("no_lifetime_span")
    return notes
