import unittest

from core.benchmarking import BenchmarkSuiteResult, BenchmarkSuiteRun
from core.ranking import get_supported_ranking_objectives, rank_benchmark_suite
from core.schemas import BenchmarkSummary


def _summary(
    *,
    spec_pass: float | None,
    rs_mae: float | None,
    thickness_mae: float | None = None,
    rsu_mae: float | None = None,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        split_type="dummy",
        folds=[],
        rs_mae_mean=rs_mae,
        thickness_mae_mean=thickness_mae,
        rsu_mae_mean=rsu_mae,
        spec_pass_accuracy_mean=spec_pass,
    )


def _suite_result() -> BenchmarkSuiteResult:
    # Two bundles, each evaluated on two split modes (aggregation across split modes).
    # Bundle A: better spec, slightly worse RS.
    # Bundle B: worse spec, better RS.
    runs = [
        BenchmarkSuiteRun("A", "split1", _summary(spec_pass=0.90, rs_mae=1.20, thickness_mae=4.0, rsu_mae=0.20)),
        BenchmarkSuiteRun("A", "split2", _summary(spec_pass=0.80, rs_mae=1.00, thickness_mae=4.2, rsu_mae=0.25)),
        BenchmarkSuiteRun("B", "split1", _summary(spec_pass=0.70, rs_mae=0.60, thickness_mae=5.0, rsu_mae=0.35)),
        BenchmarkSuiteRun("B", "split2", _summary(spec_pass=0.75, rs_mae=0.80, thickness_mae=4.8, rsu_mae=0.30)),
    ]
    return BenchmarkSuiteResult(runs=runs)


class TestRankingObjectives(unittest.TestCase):
    def test_supported_objectives_metadata(self) -> None:
        meta = get_supported_ranking_objectives()
        self.assertTrue(isinstance(meta, dict))
        for key in ("spec_pass_first", "rs_first", "thickness_first", "rsu_first", "weighted_combined"):
            self.assertIn(key, meta)
            self.assertIn("display_name", meta[key])
            self.assertIn("description", meta[key])
            self.assertIn("uses_custom_weights", meta[key])

    def test_each_objective_is_deterministic(self) -> None:
        suite = _suite_result()
        for objective in get_supported_ranking_objectives().keys():
            r1 = rank_benchmark_suite(suite, objective=objective)
            r2 = rank_benchmark_suite(suite, objective=objective)
            self.assertEqual(r1, r2)
            self.assertEqual({x["bundle_name"] for x in r1}, {"A", "B"})

    def test_spec_pass_first_prefers_higher_spec_then_lower_rs(self) -> None:
        suite = _suite_result()
        ranked = rank_benchmark_suite(suite, objective="spec_pass_first")
        self.assertEqual(ranked[0]["bundle_name"], "A")

    def test_rs_first_prefers_lower_rs_then_higher_spec(self) -> None:
        suite = _suite_result()
        ranked = rank_benchmark_suite(suite, objective="rs_first")
        self.assertEqual(ranked[0]["bundle_name"], "B")

    def test_thickness_first_prefers_lower_thickness_then_higher_spec(self) -> None:
        suite = _suite_result()
        ranked = rank_benchmark_suite(suite, objective="thickness_first")
        # A has lower thickness means than B in the synthetic setup.
        self.assertEqual(ranked[0]["bundle_name"], "A")

    def test_rsu_first_prefers_lower_rsu_then_higher_spec(self) -> None:
        suite = _suite_result()
        ranked = rank_benchmark_suite(suite, objective="rsu_first")
        self.assertEqual(ranked[0]["bundle_name"], "A")

    def test_weighted_combined_respects_weights(self) -> None:
        suite = _suite_result()

        # If we heavily weight spec-pass, A should win.
        ranked_spec = rank_benchmark_suite(
            suite,
            objective="weighted_combined",
            weights={"spec_pass_accuracy": 10.0, "rs_mae": 0.1, "thickness_mae": 0.0, "rsu_mae": 0.0},
        )
        self.assertEqual(ranked_spec[0]["bundle_name"], "A")
        self.assertIn("weighted_score", ranked_spec[0])

        # If we heavily weight RS MAE, B should win.
        ranked_rs = rank_benchmark_suite(
            suite,
            objective="weighted_combined",
            weights={"spec_pass_accuracy": 0.1, "rs_mae": 10.0, "thickness_mae": 0.0, "rsu_mae": 0.0},
        )
        self.assertEqual(ranked_rs[0]["bundle_name"], "B")
        self.assertIn("weighted_score", ranked_rs[0])

    def test_missing_metrics_do_not_crash_and_rank_worst(self) -> None:
        suite = _suite_result()
        suite2 = BenchmarkSuiteResult(
            runs=list(suite.runs)
            + [
                BenchmarkSuiteRun("C", "split1", _summary(spec_pass=None, rs_mae=None, thickness_mae=None, rsu_mae=None)),
                BenchmarkSuiteRun("C", "split2", _summary(spec_pass=None, rs_mae=None, thickness_mae=None, rsu_mae=None)),
            ]
        )

        ranked = rank_benchmark_suite(suite2, objective="spec_pass_first")
        self.assertEqual(ranked[-1]["bundle_name"], "C")

        ranked_w = rank_benchmark_suite(suite2, objective="weighted_combined")
        self.assertEqual(ranked_w[-1]["bundle_name"], "C")


if __name__ == "__main__":
    unittest.main()

