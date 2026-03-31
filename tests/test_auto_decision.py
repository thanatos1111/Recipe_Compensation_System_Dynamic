import unittest

import numpy as np
import pandas as pd

from core.auto_decision import build_auto_evaluation_plan, run_auto_model_selection
from core.auto_decision import (
    _build_explanation,
    render_auto_decision_report_text,
    score_auto_decision_candidates,
    DECISION_MODE_RECOMMENDATION_FIRST,
)
from core.benchmarking import MODEL_BUNDLE_PRESETS
from core.data_regime import DataRegime
from core.recommendation_backtest import RecommendationBacktestResult, RecommendationBacktestSummary
from core.schemas import SpecConfig


def _make_df(n_targets: int = 2, n_per_target: int = 14) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    rng = np.random.default_rng(0)
    for t in range(n_targets):
        tid = f"T{t + 1}"
        for i in range(n_per_target):
            lifetime = float(i)
            incident_angle = 10.0 + 0.1 * i
            rs = 20.0 + 0.2 * lifetime + 0.1 * incident_angle + 0.05 * t + float(rng.normal(0, 0.0))
            thickness = 1000.0 + 0.3 * lifetime - 0.05 * incident_angle + float(rng.normal(0, 0.0))
            rsu = 1.0 + 0.01 * lifetime + 0.01 * t
            rows.append(
                {
                    "target_id": tid,
                    "lifetime": lifetime,
                    "incident_angle": incident_angle,
                    "linear_offset": 0.0,
                    "rotations": 100.0,
                    "ar_flow": 10.0,
                    "o2_flow": 5.0,
                    "total_flow": 15.0,
                    "o2_ratio": 5.0 / 15.0,
                    "rpm": 10.0,
                    "power": 1000.0,
                    "rs": rs,
                    "thickness": thickness,
                    "rsu": rsu,
                    "in_spec": True,
                }
            )
    return pd.DataFrame(rows)


class TestAutoDecision(unittest.TestCase):
    def _base_config(self) -> dict:
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        return {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
                "leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3},
                # Keep uncertainty disabled by default in this test.
                "uncertainty": {"enabled": False},
                "ranking_objective": "spec_pass_first",
                "uncertainty_mode": "ignore",
            },
            # Keep backtest disabled in tests (no parameter_constraints).
            "custom_model_bundles": {},
            "auto_decision": {"bundle_allowlist": ["baseline_linear", "baseline_tree"], "shortlist_size": 2},
        }

    def test_build_plan_selects_valid_split_modes(self) -> None:
        df = _make_df(n_targets=2, n_per_target=12)
        cfg = self._base_config()
        plan = build_auto_evaluation_plan(df, cfg)
        self.assertIn("forward_chaining", plan["split_modes"])
        # With 2 targets and sufficient rows, LOTO should be feasible.
        self.assertIn("leave_one_target_out", plan["split_modes"])
        self.assertIn("baseline_linear", plan["eligible_bundles"])

    def test_sparse_data_avoids_invalid_paths(self) -> None:
        df = _make_df(n_targets=1, n_per_target=6)
        cfg = self._base_config()
        plan = build_auto_evaluation_plan(df, cfg)
        self.assertIn("forward_chaining", plan["split_modes"])
        self.assertNotIn("leave_one_target_out", plan["split_modes"])

    def test_auto_decision_returns_structured_result(self) -> None:
        df = _make_df(n_targets=2, n_per_target=14)
        cfg = self._base_config()
        # Provide preset bundle definitions explicitly to ensure stable evaluation.
        cfg["auto_decision"]["bundle_allowlist"] = ["baseline_linear", "baseline_tree"]
        out = run_auto_model_selection(df, cfg)
        self.assertIsNotNone(out.detected_regime)
        self.assertTrue(out.split_modes_used)
        self.assertTrue(out.bundles_evaluated)
        self.assertTrue(out.shortlisted_bundles)
        self.assertIn(out.winner, out.bundles_evaluated)
        self.assertIn(out.confidence_level, {"low", "moderate", "high"})
        self.assertTrue(isinstance(out.explanation_text, str) and out.explanation_text)

    def test_allowlist_filter_is_respected(self) -> None:
        df = _make_df(n_targets=2, n_per_target=14)
        cfg = self._base_config()
        cfg["auto_decision"]["bundle_allowlist"] = ["baseline_linear"]
        # Ensure the preset exists (sanity).
        self.assertIn("baseline_linear", MODEL_BUNDLE_PRESETS)
        out = run_auto_model_selection(df, cfg)
        self.assertEqual(out.bundles_evaluated, ["baseline_linear"])

    def test_recommendation_first_ranks_by_backtest_predicted_spec_improvement(self) -> None:
        ranked = [
            {"bundle_name": "A", "primary_mean": 0.1, "secondary_mean": 0.2},
            {"bundle_name": "B", "primary_mean": 0.2, "secondary_mean": 0.1},
        ]
        backtest = RecommendationBacktestResult(
            split_mode="leave_one_target_out",
            split_folds=1,
            summaries_by_bundle={
                "A": RecommendationBacktestSummary(
                    bundle_name="A",
                    rows_evaluated=20,
                    recommendation_improvement_rate=0.1,
                    predicted_spec_pass_improvement_rate=0.4,
                    no_change_fraction=0.3,
                    move_mean_abs_total=None,
                    move_median_abs_total=None,
                    move_max_abs_total=None,
                    warnings=[],
                ),
                "B": RecommendationBacktestSummary(
                    bundle_name="B",
                    rows_evaluated=20,
                    recommendation_improvement_rate=0.2,
                    predicted_spec_pass_improvement_rate=0.1,
                    no_change_fraction=0.2,
                    move_mean_abs_total=None,
                    move_median_abs_total=None,
                    move_max_abs_total=None,
                    warnings=[],
                ),
            },
            aborted=False,
        )
        scored = score_auto_decision_candidates(
            ranked_bundles=ranked,
            backtest_result=backtest,
            objective="spec_pass_first",
            decision_mode=DECISION_MODE_RECOMMENDATION_FIRST,
            backtest_min_rows_for_use=5,
        )
        self.assertEqual(scored[0]["bundle_name"], "A")

    def test_recommendation_first_falls_back_when_backtest_rows_too_low(self) -> None:
        ranked = [
            {"bundle_name": "A", "primary_mean": 0.1, "secondary_mean": 0.2},
            {"bundle_name": "B", "primary_mean": 0.2, "secondary_mean": 0.1},
        ]
        backtest = RecommendationBacktestResult(
            split_mode="leave_one_target_out",
            split_folds=1,
            summaries_by_bundle={
                "A": RecommendationBacktestSummary(
                    bundle_name="A",
                    rows_evaluated=1,
                    recommendation_improvement_rate=0.9,
                    predicted_spec_pass_improvement_rate=0.9,
                    no_change_fraction=0.1,
                    move_mean_abs_total=None,
                    move_median_abs_total=None,
                    move_max_abs_total=None,
                    warnings=[],
                ),
                "B": RecommendationBacktestSummary(
                    bundle_name="B",
                    rows_evaluated=1,
                    recommendation_improvement_rate=0.8,
                    predicted_spec_pass_improvement_rate=0.8,
                    no_change_fraction=0.2,
                    move_mean_abs_total=None,
                    move_median_abs_total=None,
                    move_max_abs_total=None,
                    warnings=[],
                ),
            },
            aborted=False,
        )
        scored = score_auto_decision_candidates(
            ranked_bundles=ranked,
            backtest_result=backtest,
            objective="spec_pass_first",
            decision_mode=DECISION_MODE_RECOMMENDATION_FIRST,
            backtest_min_rows_for_use=5,
        )
        # Benchmark order preserved: A then B.
        self.assertEqual([r["bundle_name"] for r in scored[:2]], ["A", "B"])

    def test_render_report_shows_skipped_bundles(self) -> None:
        regime = DataRegime(
            total_rows=10,
            target_count=2,
            rows_per_target={"T1": 5, "T2": 5},
            leave_one_target_out_feasible=True,
            active_target_cutoff_feasible=False,
            uncertainty_calibration_feasible=False,
            recommendation_backtest_feasible=False,
            regime_label="moderate_multi_target",
            notes=[],
        )
        from core.auto_decision import AutoDecisionResult

        out = AutoDecisionResult(
            detected_regime=regime,
            split_modes_used=["forward_chaining"],
            bundles_evaluated=["baseline_linear", "baseline_tree"],
            shortlisted_bundles=["baseline_linear"],
            winner="baseline_linear",
            runner_up=None,
            confidence_level="low",
            explanation_text="test",
            excluded_reasons={
                "split:leave_one_target_out": "not_feasible_in_regime",
                "bundle:xgb_default": "not_in_allowlist",
                "bundles:empty_after_filters": "fell_back_to_default_subset",
            },
            benchmark_suite=None,
            benchmark_ranked=None,
            backtest_result=None,
        )
        text = render_auto_decision_report_text(out)
        self.assertIn("Skipped split modes:", text)
        self.assertIn("Skipped bundles:", text)
        self.assertIn("xgb_default", text)
        self.assertIn("bundles:empty_after_filters", text)

    def test_explanation_recommendation_first_backtest_used_and_fallback(self) -> None:
        regime = DataRegime(
            total_rows=100,
            target_count=3,
            rows_per_target={"T1": 40, "T2": 30, "T3": 30},
            leave_one_target_out_feasible=True,
            active_target_cutoff_feasible=False,
            uncertainty_calibration_feasible=False,
            recommendation_backtest_feasible=True,
            regime_label="rich_multi_target",
            notes=[],
        )
        ranked = [{"bundle_name": "A", "primary_mean": 0.1, "secondary_mean": 0.2}]
        backtest = RecommendationBacktestResult(
            split_mode="leave_one_target_out",
            split_folds=1,
            summaries_by_bundle={
                "A": RecommendationBacktestSummary(
                    bundle_name="A",
                    rows_evaluated=20,
                    recommendation_improvement_rate=0.1,
                    predicted_spec_pass_improvement_rate=0.4,
                    no_change_fraction=0.3,
                    move_mean_abs_total=None,
                    move_median_abs_total=None,
                    move_max_abs_total=None,
                    warnings=[],
                )
            },
            aborted=False,
        )

        ex_used = _build_explanation(
            regime=regime,
            split_modes=["leave_one_target_out"],
            objective="spec_pass_first",
            uncertainty_mode="ignore",
            winner="A",
            runner_up=None,
            scored=ranked,
            backtest=backtest,
            decision_mode=DECISION_MODE_RECOMMENDATION_FIRST,
            used_backtest_as_primary=True,
            backtest_min_rows_for_use=5,
        )
        self.assertIn("backtest was feasible", ex_used)

        ex_fallback = _build_explanation(
            regime=regime,
            split_modes=["leave_one_target_out"],
            objective="spec_pass_first",
            uncertainty_mode="ignore",
            winner="A",
            runner_up=None,
            scored=ranked,
            backtest=None,
            decision_mode=DECISION_MODE_RECOMMENDATION_FIRST,
            used_backtest_as_primary=False,
            backtest_min_rows_for_use=5,
        )
        self.assertIn("ranking fell back to benchmark results", ex_fallback)


if __name__ == "__main__":
    unittest.main()

