import unittest

import numpy as np
import pandas as pd

from core.benchmarking import MODEL_BUNDLE_PRESETS, run_prediction_benchmark
from core.model_registry import is_model_available
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


class TestBenchmarkingExternalModels(unittest.TestCase):
    def test_external_model_missing_dependencies_do_not_crash(self) -> None:
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

        bundles = {
            "xgb_default": MODEL_BUNDLE_PRESETS["xgb_default"],
            "lgbm_default": MODEL_BUNDLE_PRESETS["lgbm_default"],
            "catboost_default": MODEL_BUNDLE_PRESETS["catboost_default"],
        }

        out = run_prediction_benchmark(
            df,
            config=cfg,
            model_names_by_target=bundles,
            split_modes=["forward_chaining"],
        )

        self.assertEqual(len(out.runs), len(bundles))

        def has_missing_warning(warnings: list[str], needle: str) -> bool:
            needle_l = needle.lower()
            return any(needle_l in (w or "").lower() and "unavailable" in (w or "").lower() for w in warnings)

        for run in out.runs:
            bn = run.bundle_name
            if bn == "xgb_default":
                missing = not is_model_available("xgb")
                if missing:
                    self.assertTrue(has_missing_warning(run.summary.aggregate_warnings, "xgb"))
                    self.assertIsNone(run.summary.rs_mae_mean)
                else:
                    self.assertIsNotNone(run.summary.rs_mae_mean)
            elif bn == "lgbm_default":
                missing = not is_model_available("lgbm")
                if missing:
                    self.assertTrue(has_missing_warning(run.summary.aggregate_warnings, "lgbm"))
                    self.assertIsNone(run.summary.rs_mae_mean)
                else:
                    self.assertIsNotNone(run.summary.rs_mae_mean)
            elif bn == "catboost_default":
                missing = not is_model_available("catboost")
                if missing:
                    self.assertTrue(has_missing_warning(run.summary.aggregate_warnings, "catboost"))
                    self.assertIsNone(run.summary.rs_mae_mean)
                else:
                    self.assertIsNotNone(run.summary.rs_mae_mean)


if __name__ == "__main__":
    unittest.main()

