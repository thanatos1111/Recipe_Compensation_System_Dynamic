import unittest

import pandas as pd

from core.evaluation_mode import ScenarioConfig, evaluate_data_sufficiency_scenarios
from core.schemas import SpecConfig


class TestEvaluationMode(unittest.TestCase):
    def _make_df(self) -> pd.DataFrame:
        rows: list[dict[str, float | str | bool]] = []
        for tid, offset in [("Ta.1", 0.0), ("Ta.2", 1.0)]:
            for i in range(30):
                lifetime = float(i)
                incident_angle = 10.0 + 0.2 * i
                linear_offset = 0.5 + 0.01 * i
                rotations = 80.0 + i
                ar_flow = 20.0
                o2_flow = 10.0
                rs = 50.0 + 0.3 * lifetime + offset
                thickness = 1000.0 + 0.6 * lifetime - offset
                rsu = 0.7 + 0.01 * lifetime
                rows.append(
                    {
                        "target_id": tid,
                        "lifetime": lifetime,
                        "incident_angle": incident_angle,
                        "linear_offset": linear_offset,
                        "rotations": rotations,
                        "rpm": 60.0,
                        "power": 1200.0,
                        "ar_flow": ar_flow,
                        "o2_flow": o2_flow,
                        "total_flow": ar_flow + o2_flow,
                        "o2_ratio": o2_flow / (ar_flow + o2_flow),
                        "rs": rs,
                        "thickness": thickness,
                        "rsu": rsu,
                        "in_spec": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_scenario_comparison_returns_metrics(self) -> None:
        df = self._make_df()
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0, "test_fraction": 0.2},
            "benchmark_settings": {
                "split_strategy": "forward_chaining",
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
            },
            "parameter_constraints": {
                "incident_angle": {"type": "continuous", "min_value": 0.0, "max_value": 90.0, "step": 0.5, "is_enabled": True},
                "linear_offset": {"type": "continuous", "min_value": -5.0, "max_value": 5.0, "step": 0.05, "is_enabled": True},
                "rotations": {"type": "integer", "min_value": 1, "max_value": 1000, "step": 1, "is_enabled": True},
                "ar_flow": {"type": "continuous", "min_value": 1.0, "max_value": 60.0, "step": 0.5, "is_enabled": True},
                "o2_flow": {"type": "continuous", "min_value": 1.0, "max_value": 60.0, "step": 0.5, "is_enabled": True},
            },
            "optimizer_weights": {"w_rs": 1.0, "w_thickness": 1.0, "w_rsu": 1.0, "w_move": 0.1, "w_margin": 0.0},
        }
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        out = evaluate_data_sufficiency_scenarios(
            df,
            spec_config=spec,
            config=cfg,
            scenario_a=ScenarioConfig(
                name="A",
                active_target_id="Ta.2",
                history_target_ids=("Ta.1",),
                active_cutoff_lifetime=10.0,
            ),
            scenario_b=ScenarioConfig(
                name="B",
                active_target_id="Ta.2",
                history_target_ids=("Ta.1",),
                active_cutoff_lifetime=10.0,
            ),
        )
        self.assertIn("scenario_a", out)
        self.assertIn("scenario_b", out)
        self.assertIn("metrics", out["scenario_a"])
        self.assertIn("rs_mae", out["scenario_a"]["metrics"])
        self.assertIn("rs_rmse", out["scenario_a"]["metrics"])
        self.assertIn("interpretation", out)

    def test_cutoff_controls_test_set(self) -> None:
        df = self._make_df()
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 1, "min_train_rows": 10, "min_test_rows": 4},
            },
        }
        spec = SpecConfig(use_rs_spec=False, use_thickness_spec=False, use_rsu_spec=False, rsu_max=0.0)
        out = evaluate_data_sufficiency_scenarios(
            df,
            spec_config=spec,
            config=cfg,
            scenario_a=ScenarioConfig(
                name="A",
                active_target_id="Ta.2",
                history_target_ids=("Ta.1",),
                active_cutoff_lifetime=20.0,
            ),
            scenario_b=ScenarioConfig(
                name="B",
                active_target_id="Ta.2",
                history_target_ids=("Ta.1",),
                active_cutoff_lifetime=20.0,
            ),
        )
        self.assertEqual(out["scenario_a"]["row_summary"]["n_test"], 9)
        self.assertEqual(out["scenario_b"]["row_summary"]["n_test"], 9)


if __name__ == "__main__":
    unittest.main()
