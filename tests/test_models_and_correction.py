import unittest

import numpy as np
import pandas as pd

from core.instance_correction import apply_instance_correction, fit_instance_bias
from core.response_models import train_material_models


class TestModelsAndCorrection(unittest.TestCase):
    def _make_df(self) -> pd.DataFrame:
        # Two target instances, monotonic lifetime.
        rng = np.random.default_rng(0)
        rows = []
        for tid, offset in [("Ta.1", 0.0), ("Ta.2", 1.0)]:
            for i in range(30):
                lifetime = float(i)
                incident_angle = 10.0 + 0.1 * i
                linear_offset = 0.5 + 0.01 * i
                rotations = 100 + i
                ar_flow = 10.0
                o2_flow = 5.0
                total_flow = ar_flow + o2_flow
                o2_ratio = o2_flow / total_flow
                # Simple synthetic outputs with target-specific bias.
                rs = 50.0 + 0.2 * lifetime + offset
                thickness = 1000.0 + 0.5 * lifetime - offset
                rsu = 0.5 + 0.01 * lifetime
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
                        "rs": rs + float(rng.normal(0, 0.2)),
                        "thickness": thickness + float(rng.normal(0, 0.5)),
                        "rsu": rsu + float(rng.normal(0, 0.01)),
                        "in_spec": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_train_material_models_and_predict_bias(self) -> None:
        df = self._make_df()
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0, "test_fraction": 0.2},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 5},
            },
        }

        artifacts = train_material_models(df, config=cfg)
        self.assertIsNotNone(artifacts)
        self.assertIn("rs_mae", artifacts.metrics)

        # Instance correction should return bias terms when using fitted models.
        model_bundle = {
            "rs_model": artifacts.rs_model,
            "thickness_model": artifacts.thickness_model,
            "rsu_model": artifacts.rsu_model,
            "feature_config": cfg.get("feature_config", {}),
        }
        inst_df = df[df["target_id"] == "Ta.2"].sort_values("lifetime")
        correction = fit_instance_bias(model_bundle, inst_df)
        self.assertIn("rs_bias", correction.bias_terms)

        # Applying correction should change predictions.
        preds = {"rs_pred": np.array([1.0, 2.0, 3.0])}
        corrected = apply_instance_correction(preds, correction)
        self.assertNotEqual(float(corrected["rs_pred"][0]), 1.0)

    def test_train_material_models_with_nonconsecutive_subset_index(self) -> None:
        df = self._make_df()
        # Simulate UI subset training where selected targets may not start from first target.
        subset = df[df["target_id"].isin(["Ta.2"])].copy()
        # Preserve non-consecutive labels to reproduce original failure mode.
        self.assertFalse(subset.index.equals(pd.RangeIndex(start=0, stop=len(subset), step=1)))

        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0, "test_fraction": 0.2},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 5},
            },
        }
        artifacts = train_material_models(subset, config=cfg)
        self.assertIsNotNone(artifacts)
        self.assertIn("rs_mae", artifacts.metrics)


if __name__ == "__main__":
    unittest.main()

