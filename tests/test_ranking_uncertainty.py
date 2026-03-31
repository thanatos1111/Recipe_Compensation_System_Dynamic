"""Tests for uncertainty-aware benchmark ranking."""

from __future__ import annotations

import unittest

from core.benchmarking import BenchmarkSuiteResult, BenchmarkSuiteRun
from core.ranking import (
    get_supported_uncertainty_modes,
    rank_benchmark_suite,
    score_benchmark_bundle,
)
from core.schemas import BenchmarkSummary


def _summary(
    *,
    spec_pass: float | None,
    rs_mae: float | None,
    rs_cov: float | None = None,
    rs_mean_w: float | None = None,
    th_cov: float | None = None,
    th_mean_w: float | None = None,
    rsu_cov: float | None = None,
    rsu_mean_w: float | None = None,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        split_type="dummy",
        folds=[],
        rs_mae_mean=rs_mae,
        thickness_mae_mean=1.0,
        rsu_mae_mean=0.1,
        spec_pass_accuracy_mean=spec_pass,
        rs_interval_coverage_mean=rs_cov,
        rs_interval_mean_width_mean=rs_mean_w,
        thickness_interval_coverage_mean=th_cov,
        thickness_interval_mean_width_mean=th_mean_w,
        rsu_interval_coverage_mean=rsu_cov,
        rsu_interval_mean_width_mean=rsu_mean_w,
    )


class TestRankingUncertainty(unittest.TestCase):
    def test_supported_uncertainty_modes(self) -> None:
        meta = get_supported_uncertainty_modes()
        for key in ("ignore", "warn_only", "include_in_score"):
            self.assertIn(key, meta)
            self.assertIn("display_name", meta[key])

    def test_missing_interval_metrics_do_not_crash(self) -> None:
        runs = [
            BenchmarkSuiteRun(
                "A",
                "s1",
                _summary(spec_pass=0.8, rs_mae=1.0, rs_cov=None, rs_mean_w=None),
            ),
            BenchmarkSuiteRun(
                "B",
                "s1",
                _summary(spec_pass=0.7, rs_mae=1.1, rs_cov=None, rs_mean_w=None),
            ),
        ]
        suite = BenchmarkSuiteResult(runs=runs)
        r_ignore = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="ignore")
        r_inc = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="include_in_score")
        self.assertEqual({x["bundle_name"] for x in r_ignore}, {"A", "B"})
        self.assertEqual({x["bundle_name"] for x in r_inc}, {"A", "B"})
        self.assertEqual(r_ignore[0]["bundle_name"], "A")
        sc = score_benchmark_bundle(
            [runs[0].summary],
            objective="spec_pass_first",
            uncertainty_mode="include_in_score",
        )
        self.assertIn("interval_aggregates", sc)
        self.assertNotIn("uncertainty_warnings", sc)

    def test_warn_only_metadata_without_rank_change(self) -> None:
        # Same primary/secondary; off-target coverage should trigger warnings for B only.
        runs = [
            BenchmarkSuiteRun(
                "A",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.9,
                    rs_mean_w=1.0,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
            BenchmarkSuiteRun(
                "B",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.5,
                    rs_mean_w=1.0,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
        ]
        suite = BenchmarkSuiteResult(runs=runs)
        r_ignore = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="ignore")
        r_warn = rank_benchmark_suite(suite, objective="spec_pass_first", uncertainty_mode="warn_only", conformal_alpha=0.1)
        self.assertEqual([x["bundle_name"] for x in r_ignore], [x["bundle_name"] for x in r_warn])
        by_name = {x["bundle_name"]: x for x in r_warn}
        self.assertTrue(any("below target" in w for w in by_name["B"].get("uncertainty_warnings", [])))
        self.assertEqual(by_name["A"].get("uncertainty_warnings", []), [])

    def test_include_in_score_prefers_calibrated_bundle_on_tie(self) -> None:
        runs = [
            BenchmarkSuiteRun(
                "miscal",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.55,
                    rs_mean_w=0.5,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
            BenchmarkSuiteRun(
                "calib",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.9,
                    rs_mean_w=0.5,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
        ]
        suite = BenchmarkSuiteResult(runs=runs)
        ranked = rank_benchmark_suite(
            suite,
            objective="spec_pass_first",
            uncertainty_mode="include_in_score",
            conformal_alpha=0.1,
        )
        self.assertEqual(ranked[0]["bundle_name"], "calib")
        self.assertGreater(float(ranked[0]["uncertainty_quality"]), float(ranked[1]["uncertainty_quality"]))

    def test_weighted_combined_boosts_calibrated_when_uncertainty_included(self) -> None:
        # Names chosen so a score tie sorts the worse bundle first when uncertainty is ignored.
        runs = [
            BenchmarkSuiteRun(
                "aaa_wide",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.6,
                    rs_mean_w=3.0,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
            BenchmarkSuiteRun(
                "zzz_tight",
                "s1",
                _summary(
                    spec_pass=0.8,
                    rs_mae=1.0,
                    rs_cov=0.9,
                    rs_mean_w=0.2,
                    th_cov=0.9,
                    th_mean_w=1.0,
                    rsu_cov=0.9,
                    rsu_mean_w=1.0,
                ),
            ),
        ]
        suite = BenchmarkSuiteResult(runs=runs)
        wts = {"spec_pass_accuracy": 1.0, "rs_mae": 1.0, "thickness_mae": 0.0, "rsu_mae": 0.0}
        r_ignore = rank_benchmark_suite(
            suite,
            objective="weighted_combined",
            weights=wts,
            uncertainty_mode="ignore",
        )
        r_inc = rank_benchmark_suite(
            suite,
            objective="weighted_combined",
            weights=wts,
            uncertainty_mode="include_in_score",
            uncertainty_weights={"interval_quality": 2.0},
            conformal_alpha=0.1,
        )
        # Same accuracy metrics -> tie on ignore (tie-break is bundle name: aaa before zzz).
        self.assertAlmostEqual(float(r_ignore[0]["weighted_score"]), float(r_ignore[1]["weighted_score"]), places=6)
        self.assertEqual(r_ignore[0]["bundle_name"], "aaa_wide")
        self.assertEqual(r_inc[0]["bundle_name"], "zzz_tight")


if __name__ == "__main__":
    unittest.main()
