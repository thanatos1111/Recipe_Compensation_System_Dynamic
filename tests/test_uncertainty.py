import unittest

import numpy as np
import pandas as pd

from core.benchmarking import MODEL_BUNDLE_PRESETS, run_prediction_benchmark
from core.schemas import SpecConfig
from core.uncertainty import (
    apply_conformal_interval,
    evaluate_interval_quality,
    fit_residual_conformal_calibrator,
)


class TestUncertainty(unittest.TestCase):
    def test_conformal_intervals_have_valid_lower_upper_ordering(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        y_pred = y_true + np.array([0.1, -0.2, 0.0, 0.4, -0.1], dtype=float)

        calibrator = fit_residual_conformal_calibrator(y_true, y_pred, alpha=0.2)
        intervals = apply_conformal_interval(y_pred, calibrator)

        self.assertIn("lower", intervals.columns)
        self.assertIn("upper", intervals.columns)
        self.assertTrue(np.all(intervals["lower"].to_numpy() <= intervals["upper"].to_numpy()))

    def test_interval_metrics_compute_correctly(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        lower = np.array([0.0, 0.0, 1.0, 2.0], dtype=float)
        upper = np.array([0.5, 2.0, 2.5, 3.5], dtype=float)

        q = evaluate_interval_quality(y_true, lower, upper)
        self.assertAlmostEqual(float(q["coverage"]), 1.0, places=12)

        # widths = [0.5, 2.0, 1.5, 1.5]
        expected_mean = (0.5 + 2.0 + 1.5 + 1.5) / 4.0  # 1.375
        expected_median = 1.5

        self.assertAlmostEqual(float(q["mean_interval_width"]), expected_mean, places=12)
        self.assertAlmostEqual(float(q["median_interval_width"]), expected_median, places=12)

    def test_benchmark_can_include_uncertainty_metrics_when_enabled(self) -> None:
        # Synthetic but "learnable" data with enough points for folds + calibration.
        rows: list[dict[str, float | str | bool]] = []
        for tid, offset in [("T1", 0.0), ("T2", 0.5), ("T3", 1.0)]:
            for i in range(12):
                lifetime = float(i)
                incident_angle = 10.0 + 0.1 * i
                rs = 20.0 + 0.8 * lifetime + 0.5 * incident_angle + offset
                thickness = 1000.0 + 2.5 * lifetime - 0.25 * incident_angle - 3.0 * offset
                rsu = 1.0 + 0.02 * lifetime + 0.05 * offset
                rows.append(
                    {
                        "target_id": tid,
                        "lifetime": lifetime,
                        "incident_angle": incident_angle,
                        "linear_offset": 0.5,
                        "rotations": 100.0,
                        "ar_flow": 10.0,
                        "o2_flow": 5.0,
                        "total_flow": 15.0,
                        "o2_ratio": 5.0 / 15.0,
                        "rs": float(rs),
                        "thickness": float(thickness),
                        "rsu": float(rsu),
                        "in_spec": True,
                    }
                )

        df = pd.DataFrame(rows)
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        cfg = {
            "spec_config": spec,
            "feature_config": {"numeric_features": ["lifetime", "incident_angle"], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 20, "min_test_rows": 4},
                "uncertainty": {
                    "enabled": True,
                    "alpha": 0.1,
                    "calibration_fraction": 0.25,
                    "min_calibration_rows": 5,
                },
            },
        }

        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target={"baseline_linear": MODEL_BUNDLE_PRESETS["baseline_linear"]},
            split_modes=["forward_chaining"],
        )

        self.assertGreaterEqual(len(out.runs), 1)
        folds = out.runs[0].summary.folds
        rs_covs = [f.rs_interval_coverage for f in folds if f.rs_interval_coverage is not None]
        rs_mean_widths = [f.rs_interval_mean_width for f in folds if f.rs_interval_mean_width is not None]

        # When uncertainty is enabled, conformal interval metrics should appear.
        self.assertTrue(rs_covs)
        self.assertTrue(rs_mean_widths)


if __name__ == "__main__":
    unittest.main()

