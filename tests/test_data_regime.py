import unittest

import pandas as pd

from core.data_regime import inspect_data_regime


def _make_df(*, targets: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tid, n in targets.items():
        for i in range(n):
            rows.append(
                {
                    "target_id": tid,
                    "lifetime": float(i),
                    "incident_angle": 10.0,
                    "linear_offset": 0.0,
                    "rotations": 100.0,
                    "ar_flow": 10.0,
                    "o2_flow": 5.0,
                }
            )
    return pd.DataFrame(rows)


class TestDataRegime(unittest.TestCase):
    def test_empty_dataset_is_very_sparse(self) -> None:
        df = pd.DataFrame()
        regime = inspect_data_regime(df, config={})
        self.assertEqual(regime.regime_label, "very_sparse")
        self.assertEqual(regime.total_rows, 0)
        self.assertFalse(regime.leave_one_target_out_feasible)

    def test_single_target_partial_label(self) -> None:
        df = _make_df(targets={"T1": 20})
        cfg = {"benchmark_settings": {"leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3}}}
        regime = inspect_data_regime(df, config=cfg)
        self.assertEqual(regime.target_count, 1)
        self.assertEqual(regime.regime_label, "single_target_partial")
        self.assertFalse(regime.leave_one_target_out_feasible)

    def test_loto_feasibility_multi_target(self) -> None:
        df = _make_df(targets={"T1": 12, "T2": 12})
        cfg = {"benchmark_settings": {"leave_one_target_out": {"min_train_rows": 10, "min_test_rows": 3}}}
        regime = inspect_data_regime(df, config=cfg)
        self.assertTrue(regime.leave_one_target_out_feasible)
        self.assertIn(regime.regime_label, {"moderate_multi_target", "rich_multi_target"})

    def test_active_target_cutoff_feasibility(self) -> None:
        df = _make_df(targets={"T1": 10, "T2": 10})
        cfg = {
            "benchmark_settings": {
                "active_target_cutoff": {
                    "active_target_id": "T1",
                    "cutoff_lifetime": 4.0,
                    "history_target_ids": ("T2",),
                }
            }
        }
        regime = inspect_data_regime(df, config=cfg)
        self.assertTrue(regime.active_target_cutoff_feasible)

    def test_uncertainty_feasibility_gate(self) -> None:
        df = _make_df(targets={"T1": 25, "T2": 25})
        cfg = {
            "benchmark_settings": {
                "uncertainty": {
                    "enabled": True,
                    "alpha": 0.1,
                    "calibration_fraction": 0.2,
                    "min_calibration_rows": 10,
                }
            }
        }
        regime = inspect_data_regime(df, config=cfg)
        self.assertTrue(regime.uncertainty_calibration_feasible)


if __name__ == "__main__":
    unittest.main()

