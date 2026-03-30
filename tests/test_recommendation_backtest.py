import unittest
from unittest.mock import patch
import threading

import numpy as np
import pandas as pd

from core.recommendation_backtest import backtest_single_fold, run_recommendation_backtest
from core.schemas import SpecConfig
from core.validation_schemes import split_strategy_to_iter


def _make_df(n_targets: int = 2, n_per_target: int = 20) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for t in range(n_targets):
        tid = f"T{t + 1}"
        for i in range(n_per_target):
            lifetime = float(i)
            # Quantize incident_angle on a 0.1 step grid to play nicely with step validation.
            incident_angle = round(0.2 + 0.1 * (i % 6), 1)
            linear_offset = 0.0
            rotations = float(10 + (i % 3))

            ar_flow = 10.0
            o2_flow = 5.0
            total_flow = ar_flow + o2_flow
            o2_ratio = o2_flow / total_flow

            rpm = 10.0
            power = 1000.0

            # Only RSU spec will be active: make RSU depend strongly on incident_angle.
            # Lower incident_angle => higher RSU => more out-of-spec.
            rsu = 1.6 - 0.5 * incident_angle

            # Provide placeholders for other targets (not used by RS/Thickness spec in test config).
            rs = 100.0 + 0.1 * lifetime
            thickness = 1000.0 + 0.2 * lifetime

            rows.append(
                {
                    "target_id": tid,
                    "lifetime": lifetime,
                    "incident_angle": incident_angle,
                    "linear_offset": linear_offset,
                    "rotations": rotations,
                    "ar_flow": ar_flow,
                    "o2_flow": o2_flow,
                    "total_flow": total_flow,
                    "o2_ratio": o2_ratio,
                    "rpm": rpm,
                    "power": power,
                    "rs": rs,
                    "thickness": thickness,
                    "rsu": rsu,
                }
            )

    df = pd.DataFrame(rows)
    # Derive in_spec according to the test spec (rsu only).
    df["in_spec"] = df["rsu"] <= 1.3
    return df


class TestRecommendationBacktest(unittest.TestCase):
    def _base_config(self) -> dict:
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=True, rsu_max=1.3)
        return {
            "spec_config": spec,
            "feature_config": {
                "numeric_features": [
                    "lifetime",
                    "incident_angle",
                    "linear_offset",
                    "rotations",
                    "rpm",
                    "power",
                    "ar_flow",
                    "o2_flow",
                    "o2_ratio",
                    "total_flow",
                ],
                "categorical_features": ["target_id"],
            },
            "model_settings": {"random_state": 0, "gbr": {}},
            "parameter_constraints": {
                "incident_angle": {
                    "type": "continuous",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.1,
                    "allowed_values": None,
                    "is_enabled": True,
                    "is_coupled": False,
                    "coupling_group": None,
                },
                "linear_offset": {
                    "type": "continuous",
                    "min_value": 0.0,
                    "max_value": 0.0,
                    "step": 0.0,
                    "allowed_values": None,
                    "is_enabled": False,
                    "is_coupled": False,
                    "coupling_group": None,
                },
                "rotations": {
                    "type": "integer",
                    "min_value": 0.0,
                    "max_value": 1000.0,
                    "step": 1.0,
                    "allowed_values": None,
                    "is_enabled": False,
                    "is_coupled": False,
                    "coupling_group": None,
                },
                "ar_flow": {
                    "type": "continuous",
                    "min_value": 0.0,
                    "max_value": 1000.0,
                    "step": 0.0,
                    "allowed_values": None,
                    "is_enabled": False,
                    "is_coupled": False,
                    "coupling_group": None,
                },
                "o2_flow": {
                    "type": "continuous",
                    "min_value": 0.0,
                    "max_value": 1000.0,
                    "step": 0.0,
                    "allowed_values": None,
                    "is_enabled": False,
                    "is_coupled": False,
                    "coupling_group": None,
                },
            },
            "optimizer_weights": {"w_rs": 1.0, "w_thickness": 1.0, "w_rsu": 1.0, "w_move": 0.1, "w_margin": 0.1},
            "benchmark_settings": {"forward_chaining": {"n_splits": 1, "min_train_rows": 15, "min_test_rows": 5}},
        }

    def test_backtest_runs_and_returns_bundle_metrics(self) -> None:
        df = _make_df(n_targets=2, n_per_target=20)
        cfg = self._base_config()

        out = run_recommendation_backtest(
            df,
            config=cfg,
            model_bundle_names=["baseline_linear"],
            split_mode="forward_chaining",
        )

        self.assertIn("baseline_linear", out.summaries_by_bundle)
        s = out.summaries_by_bundle["baseline_linear"]
        self.assertGreater(s.rows_evaluated, 0)
        self.assertIsNotNone(s.recommendation_improvement_rate)
        self.assertIsNotNone(s.predicted_spec_pass_improvement_rate)
        self.assertIsNotNone(s.no_change_fraction)
        self.assertIsNotNone(s.move_mean_abs_total)
        self.assertIsNotNone(s.move_median_abs_total)
        self.assertIsNotNone(s.move_max_abs_total)

    def test_no_future_rows_used_in_fold_training(self) -> None:
        df = _make_df(n_targets=2, n_per_target=20)
        cfg = self._base_config()

        # Compute expected first fold train indices.
        bs = cfg.get("benchmark_settings") or {}
        extra = dict(bs.get("forward_chaining", {}) or {})
        split_iter = split_strategy_to_iter(df, "forward_chaining", extra)
        first = next(iter(split_iter))
        _, _, expected_train_idx, _expected_test_idx = first
        expected_train_set = set(expected_train_idx.tolist())

        seen_train_sets: list[set[int]] = []

        def wrapped_fit(df_train, target_col, config, *, feature_matrix=None):
            # Record the index set seen by the trainer for this fold.
            seen_train_sets.append(set(df_train.index.tolist()))
            # Call the real function to keep flow intact.
            from core.response_models import fit_model_for_target as real_fit

            return real_fit(df_train, target_col, config, feature_matrix=feature_matrix)

        with patch("core.recommendation_backtest.fit_model_for_target", side_effect=wrapped_fit):
            _ = run_recommendation_backtest(
                df,
                config=cfg,
                model_bundle_names=["baseline_linear"],
                split_mode="forward_chaining",
            )

        self.assertTrue(seen_train_sets, "Expected fit_model_for_target to be called.")
        # All trainer calls should receive only the fold's train indices.
        for tr_set in seen_train_sets:
            self.assertEqual(tr_set, expected_train_set)

    def test_backtest_aggregation_matches_fold_metrics(self) -> None:
        df = _make_df(n_targets=2, n_per_target=20)
        cfg = self._base_config()

        bs = cfg.get("benchmark_settings") or {}
        extra = dict(bs.get("forward_chaining", {}) or {})
        split_iter = split_strategy_to_iter(df, "forward_chaining", extra)
        first = next(iter(split_iter))
        _, _, train_idx, test_idx = first
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        fold = backtest_single_fold(train_df, test_df, config=cfg, model_bundle_name="baseline_linear")
        rows = int(fold["rows_evaluated"])
        self.assertGreater(rows, 0)

        expected_improvement = float(fold["improvement_rows"]) / float(rows)
        expected_spec_improvement = float(fold["predicted_spec_improved_rows"]) / float(rows)
        expected_no_change = float(fold["no_change_rows"]) / float(rows)

        out = run_recommendation_backtest(
            df,
            config=cfg,
            model_bundle_names=["baseline_linear"],
            split_mode="forward_chaining",
        )
        s = out.summaries_by_bundle["baseline_linear"]
        self.assertAlmostEqual(float(s.recommendation_improvement_rate), expected_improvement, places=12)
        self.assertAlmostEqual(float(s.predicted_spec_pass_improvement_rate), expected_spec_improvement, places=12)
        self.assertAlmostEqual(float(s.no_change_fraction), expected_no_change, places=12)

    def test_backtest_respects_max_test_rows_per_fold(self) -> None:
        df = _make_df(n_targets=2, n_per_target=20)
        cfg = self._base_config()

        bs = cfg.get("benchmark_settings") or {}
        extra = dict(bs.get("forward_chaining", {}) or {})
        split_iter = split_strategy_to_iter(df, "forward_chaining", extra)
        first = next(iter(split_iter))
        _, _, train_idx, test_idx = first
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        cap = 3
        fold = backtest_single_fold(
            train_df,
            test_df,
            config=cfg,
            model_bundle_name="baseline_linear",
            max_test_rows_per_fold=cap,
        )
        self.assertLessEqual(int(fold.get("rows_evaluated") or 0), cap)

    def test_backtest_abort_event_short_circuits(self) -> None:
        df = _make_df(n_targets=2, n_per_target=20)
        cfg = self._base_config()

        bs = cfg.get("benchmark_settings") or {}
        extra = dict(bs.get("forward_chaining", {}) or {})
        split_iter = split_strategy_to_iter(df, "forward_chaining", extra)
        first = next(iter(split_iter))
        _, _, train_idx, test_idx = first
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        abort_event = threading.Event()
        abort_event.set()

        fold = backtest_single_fold(
            train_df,
            test_df,
            config=cfg,
            model_bundle_name="baseline_linear",
            abort_event=abort_event,
        )
        self.assertTrue(bool(fold.get("aborted")))
        self.assertEqual(int(fold.get("rows_evaluated") or 0), 0)


if __name__ == "__main__":
    unittest.main()

