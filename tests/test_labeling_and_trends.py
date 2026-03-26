import unittest

import pandas as pd

from core.labeling import apply_spec_labels, compute_derived_features
from core.schemas import SpecConfig
from core.trend_fitting import build_lifetime_reference_table, quantize_series_to_step


class TestLabelingAndTrends(unittest.TestCase):
    def test_apply_spec_labels_and_derived_features(self) -> None:
        df = pd.DataFrame(
            {
                "target_id": ["Ta.1", "Ta.1", "Ta.1"],
                "lifetime": [0.0, 1.0, 2.0],
                "incident_angle": [1.0, 2.0, 3.0],
                "linear_offset": [0.1, 0.2, 0.3],
                "rotations": [10, 11, 12],
                "ar_flow": [10.0, 10.0, 10.0],
                "o2_flow": [5.0, 5.0, 5.0],
                "thickness": [99.0, 101.0, 103.0],
                "rs": [9.5, 10.5, 11.3],
                "rsu": [0.1, 0.19, 0.3],
            }
        )

        spec = SpecConfig(
            rs_target=10.0,
            rs_tol=1.0,
            use_rs_target_mode=True,
            thickness_min=98.0,
            thickness_max=102.0,
            use_thickness_target_mode=False,
            rsu_max=0.2,
        )

        labeled = compute_derived_features(df, spec)
        self.assertIn("in_spec", labeled.columns)

        self.assertEqual(labeled["in_spec"].tolist(), [True, True, False])

        # total_flow + o2_ratio derived.
        self.assertTrue("total_flow" in labeled.columns)
        self.assertTrue("o2_ratio" in labeled.columns)
        self.assertAlmostEqual(float(labeled.loc[0, "total_flow"]), 15.0, places=6)
        self.assertAlmostEqual(float(labeled.loc[0, "o2_ratio"]), 5.0 / 15.0, places=6)

        # rs_error derived in RS target mode.
        self.assertAlmostEqual(float(labeled.loc[0, "rs_error"]), -0.5, places=6)
        self.assertAlmostEqual(float(labeled.loc[0, "abs_rs_error"]), 0.5, places=6)

        # thickness_error should be NA because thickness target mode is off.
        self.assertTrue(pd.isna(labeled.loc[0, "thickness_error"]))

        # delta columns exist.
        self.assertIn("delta_incident_angle", labeled.columns)
        self.assertTrue(pd.isna(labeled.loc[0, "delta_incident_angle"]))
        self.assertAlmostEqual(float(labeled.loc[1, "delta_incident_angle"]), 1.0, places=6)

    def test_quantize_series_to_step(self) -> None:
        values = pd.Series([1.1, 1.4, 1.6])
        q = quantize_series_to_step(values, step=0.5, min_val=0.0, max_val=10.0)
        self.assertEqual(q.tolist(), [1.0, 1.5, 1.5])

    def test_build_lifetime_reference_table_in_spec_filtering_and_quantization(self) -> None:
        df = pd.DataFrame(
            {
                "lifetime": [0.0, 1.0, 2.0, 3.0],
                "in_spec": [True, False, True, True],
                "incident_angle": [1.1, 2.0, 1.4, 1.6],
                "linear_offset": [0.0, 0.0, 0.0, 0.0],
                "rotations": [10, 99, 11, 12],
                "ar_flow": [10.0, 20.0, 10.0, 10.0],
                "o2_flow": [5.0, 10.0, 5.0, 5.0],
                "rs": [10.0, 999.0, 11.0, 12.0],
                "thickness": [100.0, 888.0, 101.0, 102.0],
                "rsu": [0.1, 9.9, 0.2, 0.3],
            }
        )

        cfg = {
            "fit_settings": {"lifetime_bin_count": 2},
            "parameter_constraints": {
                "incident_angle": {"step": 0.5, "min_value": None, "max_value": None},
                "linear_offset": {"step": 0.0, "min_value": None, "max_value": None},
                "rotations": {"step": 1.0, "min_value": None, "max_value": None},
                "ar_flow": {"step": 0.0, "min_value": None, "max_value": None},
                "o2_flow": {"step": 0.0, "min_value": None, "max_value": None},
            },
        }

        baseline = build_lifetime_reference_table(df, cfg)
        self.assertEqual(len(baseline), 2)

        # Bin 0 covers lifetime [0, 1.5] -> rows at 0.0 (in-spec) and 1.0 (out-of-spec).
        self.assertEqual(int(baseline.iloc[0]["total_count"]), 2)
        self.assertEqual(int(baseline.iloc[0]["in_spec_count"]), 1)
        self.assertAlmostEqual(float(baseline.iloc[0]["in_spec_fraction"]), 0.5, places=6)
        self.assertAlmostEqual(float(baseline.iloc[0]["incident_angle_rec"]), 1.0, places=6)

        # Bin 1 covers lifetime [1.5, 3] -> rows at 2.0 and 3.0, both in-spec.
        self.assertEqual(int(baseline.iloc[1]["total_count"]), 2)
        self.assertEqual(int(baseline.iloc[1]["in_spec_count"]), 2)
        self.assertAlmostEqual(float(baseline.iloc[1]["in_spec_fraction"]), 1.0, places=6)
        self.assertAlmostEqual(float(baseline.iloc[1]["incident_angle_rec"]), 1.5, places=6)

    def test_apply_spec_labels_rs_range_partial_bounds(self) -> None:
        df = pd.DataFrame(
            {
                "target_id": ["Ta.1", "Ta.1", "Ta.1"],
                "lifetime": [0.0, 1.0, 2.0],
                "rs": [5.0, 50.0, 500.0],
                "thickness": [100.0, 100.0, 100.0],
                "rsu": [0.1, 0.1, 0.1],
                "incident_angle": [1.0, 1.0, 1.0],
                "linear_offset": [0.1, 0.1, 0.1],
                "rotations": [10, 10, 10],
                "ar_flow": [10.0, 10.0, 10.0],
                "o2_flow": [5.0, 5.0, 5.0],
            }
        )

        # Range mode with min unset and max=100 => rs must be <= 100.
        spec = SpecConfig(
            rs_min=None,
            rs_max=100.0,
            use_rs_target_mode=False,
            rsu_max=0.2,
            use_thickness_target_mode=False,
            thickness_min=None,
            thickness_max=None,
            use_rs_spec=True,
            use_thickness_spec=False,  # ignore thickness for this test
            use_rsu_spec=True,
        )

        labeled = apply_spec_labels(df, spec)
        self.assertEqual(labeled["in_spec"].tolist(), [True, True, False])

        # Range mode with max unset and min=100 => rs must be >= 100.
        spec2 = SpecConfig(
            rs_min=100.0,
            rs_max=None,
            use_rs_target_mode=False,
            rsu_max=0.2,
            use_thickness_target_mode=False,
            thickness_min=None,
            thickness_max=None,
            use_rs_spec=True,
            use_thickness_spec=False,
            use_rsu_spec=True,
        )
        labeled2 = apply_spec_labels(df, spec2)
        self.assertEqual(labeled2["in_spec"].tolist(), [False, False, True])

    def test_fit_parameter_trend_quantization(self) -> None:
        from core.trend_fitting import fit_parameter_trend

        df = pd.DataFrame(
            {
                "lifetime": [0.0, 1.0, 2.0, 3.0],
                "in_spec": [True, True, True, True],
                "incident_angle": [1.1, 1.2, 1.3, 1.4],
            }
        )

        fitted = fit_parameter_trend(
            df,
            parameter_name="incident_angle",
            method="windowed_median",
            step=0.5,
        )

        self.assertEqual(len(fitted), 2)
        self.assertAlmostEqual(float(fitted.iloc[0]["incident_angle_ref"]), 1.0, places=6)
        self.assertAlmostEqual(float(fitted.iloc[1]["incident_angle_ref"]), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()

