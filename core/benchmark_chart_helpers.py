"""
Helpers for benchmark comparison charts (UI and future automation).

Pure functions for detecting available metrics and drawing standard plots.
"""

from __future__ import annotations

from typing import Any

# Keys on benchmark summary table rows (from ``BenchmarkSuiteResult.to_table_rows``).
_UNCERTAINTY_METRIC_KEYS: tuple[str, ...] = (
    "rs_interval_coverage_mean",
    "thickness_interval_coverage_mean",
    "rsu_interval_coverage_mean",
    "rs_interval_mean_width_mean",
    "thickness_interval_mean_width_mean",
    "rsu_interval_mean_width_mean",
    "rs_interval_median_width_mean",
    "thickness_interval_median_width_mean",
    "rsu_interval_median_width_mean",
)


def benchmark_rows_have_any_uncertainty_metrics(rows: list[dict[str, Any]]) -> bool:
    """Return True if any row carries at least one non-null uncertainty aggregate."""
    for row in rows:
        for key in _UNCERTAINTY_METRIC_KEYS:
            if row.get(key) is not None:
                return True
    return False


def uncertainty_chart_placeholder_text(*, uncertainty_enabled_for_run: bool, metrics_available: bool) -> str:
    """
    User-visible message when the Uncertainty chart tab cannot render interval plots.

    Returns an empty string when charts should be drawn instead.
    """
    if not uncertainty_enabled_for_run:
        return (
            'Enable "Enable uncertainty (conformal intervals)" in the benchmark setup above, '
            "then run the benchmark again to compare interval coverage and mean width across bundles."
        )
    if not metrics_available:
        return (
            "Uncertainty was enabled, but no interval coverage or width metrics are present in these results. "
            "This can happen if folds were too small or evaluation skipped interval metrics."
        )
    return ""


def recommendation_backtest_has_chart_data(result: Any) -> bool:
    """True if ``result.to_table_rows()`` is non-empty (chartable backtest summaries)."""
    if result is None:
        return False
    rows_fn = getattr(result, "to_table_rows", None)
    if not callable(rows_fn):
        return False
    try:
        rows = rows_fn()
    except Exception:
        return False
    return bool(rows)


def backtest_chart_placeholder_text(has_chart_data: bool) -> str:
    """Message for the Backtest benchmark tab when there is nothing to plot."""
    if has_chart_data:
        return ""
    return (
        'No recommendation backtest results to chart. Enable "Enable recommendation backtest after benchmark", '
        "run the benchmark, and wait for the backtest to finish to see improvement rates by bundle."
    )


def plot_recommendation_backtest_rates(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    title: str = "Recommendation backtest (fold replay)",
) -> None:
    """
    Draw grouped bars: recommendation improvement rate vs predicted spec-pass improvement rate per bundle.
    """
    bundles = [str(r.get("bundle_name", "")) for r in rows]
    improve = [float(r.get("recommendation_improvement_rate") or 0.0) for r in rows]
    spec_improve = [float(r.get("predicted_spec_pass_improvement_rate") or 0.0) for r in rows]

    x = list(range(len(bundles)))
    width = 0.38
    ax.bar(
        [v - width / 2 for v in x],
        improve,
        width=width,
        label="Score improvement rate",
        color="#4c78a8",
    )
    ax.bar(
        [v + width / 2 for v in x],
        spec_improve,
        width=width,
        label="Pred spec-pass improvement rate",
        color="#59a14f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bundles, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=8, loc="best")


# Tab order must match ``ModelPanel`` benchmark chart tab bar.
BENCH_CHART_TAB_SPEC_PASS = 0
BENCH_CHART_TAB_RS = 1
BENCH_CHART_TAB_THICKNESS = 2
BENCH_CHART_TAB_RSU = 3
BENCH_CHART_TAB_UNCERTAINTY = 4
BENCH_CHART_TAB_BACKTEST = 5
