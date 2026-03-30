"""
Reusable benchmark runner for leakage-free evaluation.

This is the "front door" for running benchmarks across model bundles and split
modes, so future ML methods can plug into the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.response_models import evaluate_models_with_splits
from core.schemas import BenchmarkFoldResult, BenchmarkSummary, SpecConfig


MODEL_BUNDLE_PRESETS: dict[str, dict[str, str]] = {
    # rs / thickness / rsu model names
    "baseline_linear": {"rs": "linear", "thickness": "linear", "rsu": "linear"},
    "baseline_tree": {"rs": "rf", "thickness": "rf", "rsu": "rf"},
    "balanced_default": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"},
    "mlp_experimental": {"rs": "mlp", "thickness": "mlp", "rsu": "mlp"},
}


def _normalize_model_names_by_target(model_names_by_target: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Normalize supported input shapes:

    - { "bundle_a": {"rs": "...", "thickness": "...", "rsu": "..."}, ... }
    - {"rs": "...", "thickness": "...", "rsu": "..."}  -> single bundle named "custom"
    """
    if not model_names_by_target:
        return {"custom": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"}}

    # If it already looks like a target map, wrap it.
    if all(k in model_names_by_target for k in ("rs", "thickness", "rsu")) and all(
        isinstance(model_names_by_target[k], str) for k in ("rs", "thickness", "rsu")
    ):
        return {"custom": {"rs": str(model_names_by_target["rs"]), "thickness": str(model_names_by_target["thickness"]), "rsu": str(model_names_by_target["rsu"])}}

    # Otherwise treat it as bundle_name -> target map.
    out: dict[str, dict[str, str]] = {}
    for bundle_name, target_map in model_names_by_target.items():
        if not isinstance(target_map, dict):
            raise TypeError("Each model bundle must map to a dict with keys: rs, thickness, rsu.")
        for k in ("rs", "thickness", "rsu"):
            if k not in target_map:
                raise ValueError(f"Model bundle {bundle_name!r} is missing key {k!r}.")
        out[str(bundle_name)] = {
            "rs": str(target_map["rs"]),
            "thickness": str(target_map["thickness"]),
            "rsu": str(target_map["rsu"]),
        }
    return out


def _rebuild_benchmark_summary(evaluate_out: dict[str, Any]) -> BenchmarkSummary:
    summary_dict = evaluate_out.get("summary") or {}
    folds_dicts = evaluate_out.get("folds") or summary_dict.get("folds") or []
    folds = [BenchmarkFoldResult(**fd) for fd in folds_dicts]

    return BenchmarkSummary(
        split_type=str(summary_dict.get("split_type") or evaluate_out.get("benchmark_split_type")),
        folds=folds,
        model_names=summary_dict.get("model_names") or {},
        rs_mae_mean=summary_dict.get("rs_mae_mean"),
        rs_mae_std=summary_dict.get("rs_mae_std"),
        rs_rmse_mean=summary_dict.get("rs_rmse_mean"),
        rs_rmse_std=summary_dict.get("rs_rmse_std"),
        thickness_mae_mean=summary_dict.get("thickness_mae_mean"),
        thickness_mae_std=summary_dict.get("thickness_mae_std"),
        thickness_rmse_mean=summary_dict.get("thickness_rmse_mean"),
        thickness_rmse_std=summary_dict.get("thickness_rmse_std"),
        rsu_mae_mean=summary_dict.get("rsu_mae_mean"),
        rsu_mae_std=summary_dict.get("rsu_mae_std"),
        rsu_rmse_mean=summary_dict.get("rsu_rmse_mean"),
        rsu_rmse_std=summary_dict.get("rsu_rmse_std"),
        spec_pass_accuracy_mean=summary_dict.get("spec_pass_accuracy_mean"),
        spec_pass_accuracy_std=summary_dict.get("spec_pass_accuracy_std"),
        aggregate_warnings=summary_dict.get("aggregate_warnings") or [],
    )


def benchmark_single_model_bundle(
    df,
    *,
    config: dict[str, Any],
    rs_model_name: str,
    thickness_model_name: str,
    rsu_model_name: str,
    split_mode: str,
) -> BenchmarkSummary:
    """Benchmark one rs/thickness/rsu model bundle under one split mode."""
    spec_config = config.get("spec_config")
    if spec_config is not None and not isinstance(spec_config, SpecConfig):
        raise TypeError("config['spec_config'] must be a SpecConfig (or None).")

    # Update only the per-target model selection keys.
    new_config = dict(config)
    ms = dict(new_config.get("model_settings", {}) or {})
    ms["rs_model"] = rs_model_name
    ms["thickness_model"] = thickness_model_name
    ms["rsu_model"] = rsu_model_name
    new_config["model_settings"] = ms

    eval_out = evaluate_models_with_splits(
        df,
        config=new_config,
        split_strategy=split_mode,
        spec_config=spec_config,
    )
    return _rebuild_benchmark_summary(eval_out)


@dataclass(frozen=True)
class BenchmarkSuiteRun:
    bundle_name: str
    split_mode: str
    summary: BenchmarkSummary


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    runs: list[BenchmarkSuiteRun]

    def to_table_rows(self) -> list[dict[str, Any]]:
        """UI-friendly flattened rows."""
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            rows.append(
                {
                    "bundle_name": run.bundle_name,
                    "split_mode": run.split_mode,
                    "fold_count": len(run.summary.folds),
                    "avg_train_row_count": float(
                        sum(f.train_row_count for f in run.summary.folds) / max(len(run.summary.folds), 1)
                    )
                    if run.summary.folds
                    else 0.0,
                    "avg_test_row_count": float(
                        sum(f.test_row_count for f in run.summary.folds) / max(len(run.summary.folds), 1)
                    )
                    if run.summary.folds
                    else 0.0,
                    "rs_mae_mean": run.summary.rs_mae_mean,
                    "rs_mae_std": run.summary.rs_mae_std,
                    "thickness_mae_mean": run.summary.thickness_mae_mean,
                    "thickness_mae_std": run.summary.thickness_mae_std,
                    "rsu_mae_mean": run.summary.rsu_mae_mean,
                    "rsu_mae_std": run.summary.rsu_mae_std,
                    "rs_rmse_mean": run.summary.rs_rmse_mean,
                    "rs_rmse_std": run.summary.rs_rmse_std,
                    "thickness_rmse_mean": run.summary.thickness_rmse_mean,
                    "thickness_rmse_std": run.summary.thickness_rmse_std,
                    "rsu_rmse_mean": run.summary.rsu_rmse_mean,
                    "rsu_rmse_std": run.summary.rsu_rmse_std,
                    "spec_pass_accuracy_mean": run.summary.spec_pass_accuracy_mean,
                    "spec_pass_accuracy_std": run.summary.spec_pass_accuracy_std,
                    "warnings": list(run.summary.aggregate_warnings or []) + [w for f in run.summary.folds for w in (f.warnings or [])],
                }
            )
        return rows


def run_prediction_benchmark(
    df,
    *,
    config: dict[str, Any],
    model_names_by_target: dict[str, Any],
    split_modes: list[str],
) -> BenchmarkSuiteResult:
    """Run benchmark suites across multiple model bundles and split modes."""
    bundle_map = _normalize_model_names_by_target(model_names_by_target)

    runs: list[BenchmarkSuiteRun] = []
    for bundle_name, name_map in bundle_map.items():
        for split_mode in split_modes:
            summary = benchmark_single_model_bundle(
                df,
                config=config,
                rs_model_name=name_map["rs"],
                thickness_model_name=name_map["thickness"],
                rsu_model_name=name_map["rsu"],
                split_mode=split_mode,
            )
            runs.append(BenchmarkSuiteRun(bundle_name=bundle_name, split_mode=split_mode, summary=summary))

    return BenchmarkSuiteResult(runs=runs)


def _get_metric_value(summary: BenchmarkSummary, metric: str) -> Optional[float]:
    if metric == "spec_pass_accuracy":
        return summary.spec_pass_accuracy_mean
    if metric == "rs_mae":
        return summary.rs_mae_mean
    if metric == "thickness_mae":
        return summary.thickness_mae_mean
    if metric == "rsu_mae":
        return summary.rsu_mae_mean
    # Allow direct pass-through naming for future extension.
    if metric.endswith("_mae"):
        resp = metric[: -len("_mae")]
        if resp == "rs":
            return summary.rs_mae_mean
        if resp == "thickness":
            return summary.thickness_mae_mean
        if resp == "rsu":
            return summary.rsu_mae_mean
    raise ValueError(f"Unknown metric: {metric!r}")


def rank_benchmark_results(
    results: BenchmarkSuiteResult,
    primary_metric: str = "spec_pass_accuracy",
    secondary_metric: str = "rs_mae",
) -> list[dict[str, Any]]:
    """Rank model bundles using aggregated (mean) metrics across split modes."""

    # Aggregate per bundle across all split modes requested.
    by_bundle: dict[str, list[BenchmarkSummary]] = {}
    for run in results.runs:
        by_bundle.setdefault(run.bundle_name, []).append(run.summary)

    ranked: list[dict[str, Any]] = []
    for bundle_name, summaries in by_bundle.items():
        primary_vals = [_get_metric_value(s, primary_metric) for s in summaries]
        secondary_vals = [_get_metric_value(s, secondary_metric) for s in summaries]

        def mean_ignore_none(xs: list[Optional[float]]) -> Optional[float]:
            xs2 = [float(v) for v in xs if v is not None]
            if not xs2:
                return None
            return float(sum(xs2) / len(xs2))

        ranked.append(
            {
                "bundle_name": bundle_name,
                "primary_mean": mean_ignore_none(primary_vals),
                "secondary_mean": mean_ignore_none(secondary_vals),
            }
        )

    # spec_pass_accuracy: higher is better; mae: lower is better.
    def sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
        p = item.get("primary_mean")
        s = item.get("secondary_mean")
        p_val = float(p) if p is not None else float("-inf")
        # Make sure None secondary goes to the end by treating it as +inf.
        s_val = float(s) if s is not None else float("inf")

        # We want descending primary, ascending secondary.
        return (-p_val, s_val, str(item.get("bundle_name")))

    ranked_sorted = sorted(ranked, key=sort_key)
    return ranked_sorted

