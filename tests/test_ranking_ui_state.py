"""Tests for benchmark ranking UI wiring helpers (no Qt required)."""

from __future__ import annotations

import unittest

from core.benchmark_ranking_display import format_uncertainty_ranking_explanation_lines
from core.benchmarking import BenchmarkSuiteResult, BenchmarkSuiteRun
from core.ranking import rank_benchmark_suite
from core.schemas import BenchmarkSummary


def _summary(spec_pass: float, rs_mae: float, *, rs_cov: float | None = None) -> BenchmarkSummary:
    return BenchmarkSummary(
        split_type="x",
        folds=[],
        spec_pass_accuracy_mean=spec_pass,
        rs_mae_mean=rs_mae,
        thickness_mae_mean=1.0,
        rsu_mae_mean=0.1,
        rs_interval_coverage_mean=rs_cov,
        rs_interval_mean_width_mean=1.0,
    )


class TestRankingUiState(unittest.TestCase):
    def test_explanation_includes_uncertainty_ranking_mode(self) -> None:
        ranked = [{"bundle_name": "A", "uncertainty_quality": 0.9}]
        lines = format_uncertainty_ranking_explanation_lines(
            ranked,
            uncertainty_mode="include_in_score",
            uncertainty_weights={"interval_quality": 1.0},
            uncertainty_metrics_enabled=True,
            ranking_objective="weighted_combined",
        )
        text = "\n".join(lines)
        self.assertIn("Uncertainty ranking mode: include_in_score", text)
        self.assertIn("interval_quality weight: 1.000", text)
        self.assertIn("Winner uncertainty quality: 0.9", text)

    def test_warn_only_missing_metrics_no_crash(self) -> None:
        suite = BenchmarkSuiteResult(
            runs=[
                BenchmarkSuiteRun("A", "s1", _summary(0.8, 1.0, rs_cov=None)),
            ]
        )
        ranked = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="warn_only")
        self.assertEqual(len(ranked), 1)
        lines = format_uncertainty_ranking_explanation_lines(
            ranked,
            uncertainty_mode="warn_only",
            uncertainty_metrics_enabled=False,
        )
        self.assertTrue(any("warn_only" in line for line in lines))

    def test_include_in_score_uncertainty_disabled_no_crash(self) -> None:
        suite = BenchmarkSuiteResult(
            runs=[
                BenchmarkSuiteRun("A", "s1", _summary(0.9, 1.0, rs_cov=None)),
                BenchmarkSuiteRun("B", "s1", _summary(0.7, 1.1, rs_cov=None)),
            ]
        )
        ranked = rank_benchmark_suite(
            suite,
            objective="spec_pass_first",
            uncertainty_mode="include_in_score",
            conformal_alpha=None,
        )
        self.assertEqual(len(ranked), 2)

    def test_rank_accepts_selected_uncertainty_mode(self) -> None:
        suite = BenchmarkSuiteResult(
            runs=[
                BenchmarkSuiteRun("A", "s1", _summary(0.8, 1.0, rs_cov=0.9)),
                BenchmarkSuiteRun("B", "s1", _summary(0.8, 1.0, rs_cov=0.5)),
            ]
        )
        r0 = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="ignore")
        r1 = rank_benchmark_suite(
            suite,
            objective="spec_pass_first",
            uncertainty_mode="include_in_score",
            conformal_alpha=0.1,
        )
        self.assertEqual({x["bundle_name"] for x in r0}, {"A", "B"})
        self.assertEqual(r1[0]["bundle_name"], "A")


if __name__ == "__main__":
    unittest.main()
