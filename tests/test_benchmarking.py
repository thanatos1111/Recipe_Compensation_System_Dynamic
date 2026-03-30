import unittest

import numpy as np
import pandas as pd

from core.benchmarking import (
    MODEL_BUNDLE_PRESETS,
    flatten_benchmark_suite_folds,
    rank_benchmark_results,
    run_prediction_benchmark,
)
from core.schemas import SpecConfig


def _make_df(n_targets: int = 3, n_per_target: int = 12) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    rng = np.random.default_rng(123)

    for ti in range(n_targets):
        tid = f"T{ti + 1}"
        tid_bias = 0.2 * ti
        thick_bias = 3.0 * ti
        rsu_bias = 0.05 * ti

        for i in range(n_per_target):
            lifetime = float(i)
            incident_angle = 10.0 + 0.1 * i
            linear_offset = 0.5 + 0.01 * i
            rotations = 100.0 + i
            ar_flow = 10.0
            o2_flow = 5.0
            total_flow = ar_flow + o2_flow
            o2_ratio = o2_flow / total_flow

            # Deterministic synthetic targets.
            rs = 20.0 + 0.8 * lifetime + 0.5 * incident_angle + tid_bias
            thickness = 1000.0 + 2.5 * lifetime - 0.25 * incident_angle - thick_bias
            rsu = 1.0 + 0.02 * lifetime + rsu_bias

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
                    "rs": float(rs + rng.normal(0, 0.0)),
                    "thickness": float(thickness + rng.normal(0, 0.0)),
                    "rsu": float(rsu + rng.normal(0, 0.0)),
                    "in_spec": True,
                }
            )
    return pd.DataFrame(rows)


class TestBenchmarking(unittest.TestCase):
    def test_multiple_model_bundles_run_without_crashing(self) -> None:
        df = _make_df()
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
                "leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3},
            },
        }

        model_bundles = {
            "baseline_linear": MODEL_BUNDLE_PRESETS["baseline_linear"],
            "baseline_tree": MODEL_BUNDLE_PRESETS["baseline_tree"],
        }
        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target=model_bundles,
            split_modes=["forward_chaining", "leave_one_target_out"],
        )
        self.assertEqual(len(out.runs), 4)

    def test_ranking_returns_deterministic_order(self) -> None:
        df = _make_df()
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
                "leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3},
            },
        }

        model_bundles = {
            "baseline_linear": MODEL_BUNDLE_PRESETS["baseline_linear"],
            "baseline_tree": MODEL_BUNDLE_PRESETS["baseline_tree"],
        }
        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target=model_bundles,
            split_modes=["forward_chaining", "leave_one_target_out"],
        )

        ranks1 = rank_benchmark_results(out, primary_metric="spec_pass_accuracy", secondary_metric="rs_mae")
        ranks2 = rank_benchmark_results(out, primary_metric="spec_pass_accuracy", secondary_metric="rs_mae")
        self.assertEqual(ranks1, ranks2)
        self.assertEqual({r["bundle_name"] for r in ranks1}, set(model_bundles.keys()))

    def test_split_mode_labels_are_preserved(self) -> None:
        df = _make_df()
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
                "leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3},
            },
        }

        model_bundles = {"balanced_default": MODEL_BUNDLE_PRESETS["balanced_default"]}
        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target=model_bundles,
            split_modes=["forward_chaining", "active_target_cutoff"],
        )

        modes = {run.split_mode for run in out.runs}
        self.assertEqual(modes, {"forward_chaining", "active_target_cutoff"})

    def test_fold_aggregation_is_correct(self) -> None:
        df = _make_df()
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
            },
        }

        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target={"baseline_linear": MODEL_BUNDLE_PRESETS["baseline_linear"]},
            split_modes=["forward_chaining"],
        )
        self.assertEqual(len(out.runs), 1)
        run = out.runs[0]
        self.assertGreater(len(run.summary.folds), 0)

        fold_rs_maes = [f.rs_mae for f in run.summary.folds if f.rs_mae is not None]
        self.assertTrue(fold_rs_maes)
        expected_mean = float(np.asarray(fold_rs_maes, dtype=float).mean())
        self.assertAlmostEqual(float(run.summary.rs_mae_mean), expected_mean, places=12)

    def test_flatten_fold_details(self) -> None:
        df = _make_df()
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {"forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4}},
        }

        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target={"baseline_linear": MODEL_BUNDLE_PRESETS["baseline_linear"]},
            split_modes=["forward_chaining"],
        )
        rows = flatten_benchmark_suite_folds(out)
        self.assertGreaterEqual(len(rows), len(out.runs[0].summary.folds))
        # Ensure labels exist.
        for r in rows:
            self.assertIn(r["split_mode"], {"forward_chaining"})
            self.assertTrue(isinstance(r["warnings"], str))


if __name__ == "__main__":
    unittest.main()

