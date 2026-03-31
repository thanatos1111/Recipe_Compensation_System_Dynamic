"""Tests for benchmark chart helper state/formatting logic."""

from __future__ import annotations

import unittest
from core.benchmark_chart_helpers import (
    backtest_chart_placeholder_text,
    benchmark_rows_have_any_uncertainty_metrics,
    recommendation_backtest_has_chart_data,
    uncertainty_chart_placeholder_text,
)


class TestBenchmarkChartHelpers(unittest.TestCase):
    def test_uncertainty_metrics_detection_empty(self) -> None:
        self.assertFalse(benchmark_rows_have_any_uncertainty_metrics([]))
        self.assertFalse(benchmark_rows_have_any_uncertainty_metrics([{"bundle_name": "a"}]))

    def test_uncertainty_metrics_detection_some(self) -> None:
        rows = [{"bundle_name": "a", "rs_interval_coverage_mean": 0.9}]
        self.assertTrue(benchmark_rows_have_any_uncertainty_metrics(rows))

    def test_uncertainty_placeholder_disabled(self) -> None:
        text = uncertainty_chart_placeholder_text(uncertainty_enabled_for_run=False, metrics_available=False)
        self.assertIn("conformal", text.lower())
        self.assertNotEqual(text.strip(), "")

    def test_uncertainty_placeholder_enabled_no_metrics(self) -> None:
        text = uncertainty_chart_placeholder_text(uncertainty_enabled_for_run=True, metrics_available=False)
        self.assertIn("no interval", text.lower())

    def test_uncertainty_placeholder_ready(self) -> None:
        text = uncertainty_chart_placeholder_text(uncertainty_enabled_for_run=True, metrics_available=True)
        self.assertEqual(text, "")

    def test_backtest_has_chart_data(self) -> None:
        self.assertFalse(recommendation_backtest_has_chart_data(None))

        class _Err:
            def to_table_rows(self) -> list[dict[str, str]]:
                msg = "nope"
                raise RuntimeError(msg)

        self.assertFalse(recommendation_backtest_has_chart_data(_Err()))

        class _Ok:
            def to_table_rows(self) -> list[dict[str, str]]:
                return [{"bundle_name": "x"}]

        self.assertTrue(recommendation_backtest_has_chart_data(_Ok()))

    def test_backtest_placeholder(self) -> None:
        self.assertEqual(backtest_chart_placeholder_text(True), "")
        self.assertNotEqual(backtest_chart_placeholder_text(False).strip(), "")


if __name__ == "__main__":
    unittest.main()
