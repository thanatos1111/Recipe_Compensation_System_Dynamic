import unittest

import pandas as pd

from core.recommendation_visualization import build_recommendation_visualization_payload


class TestRecommendationVisualization(unittest.TestCase):
    def test_build_payload_includes_best_selected_and_deltas(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "score": 0.10,
                    "incident_angle": 10.0,
                    "linear_offset": 0.10,
                    "rotations": 100.0,
                    "ar_flow": 20.0,
                    "o2_flow": 10.0,
                    "rs_pred": 11.0,
                    "thickness_pred": 500.0,
                    "rsu_pred": 2.0,
                    "rs_error_norm": 0.2,
                    "thickness_error_norm": 0.1,
                    "rsu_penalty": 0.0,
                    "move_penalty": 0.05,
                    "safe_margin_reward": 0.3,
                },
                {
                    "score": 0.12,
                    "incident_angle": 11.0,
                    "linear_offset": 0.11,
                    "rotations": 101.0,
                    "ar_flow": 20.5,
                    "o2_flow": 9.5,
                    "rs_pred": 11.5,
                    "thickness_pred": 498.0,
                    "rsu_pred": 2.3,
                    "rs_error_norm": 0.25,
                    "thickness_error_norm": 0.12,
                    "rsu_penalty": 0.0,
                    "move_penalty": 0.08,
                    "safe_margin_reward": 0.2,
                },
            ]
        )

        payload = build_recommendation_visualization_payload(
            ranked,
            top_n=2,
            selected_index=1,
            parameter_order=["incident_angle", "linear_offset", "rotations", "ar_flow", "o2_flow"],
        )

        self.assertEqual(payload["top_n_count"], 2)
        self.assertEqual(payload["selected_index"], 1)
        self.assertAlmostEqual(float(payload["best"]["score"]), 0.10, places=8)
        self.assertAlmostEqual(float(payload["selected"]["score"]), 0.12, places=8)

        breakdown = payload["score_breakdown"]
        self.assertIn("score", breakdown)
        self.assertIn("move_penalty", breakdown)

        deltas = payload["parameter_prediction_deltas"]
        by_param = {d["parameter"]: d for d in deltas}
        self.assertAlmostEqual(float(by_param["incident_angle"]["delta_parameter"]), 1.0, places=8)
        self.assertAlmostEqual(float(by_param["incident_angle"]["delta_rs_pred"]), 0.5, places=8)
        self.assertAlmostEqual(float(by_param["incident_angle"]["delta_thickness_pred"]), -2.0, places=8)
        self.assertAlmostEqual(float(by_param["incident_angle"]["delta_rsu_pred"]), 0.3, places=8)

    def test_empty_ranked_dataframe_returns_empty_payload(self) -> None:
        payload = build_recommendation_visualization_payload(pd.DataFrame())
        self.assertEqual(payload["top_n_count"], 0)
        self.assertEqual(payload["top_n_table"], [])
        self.assertEqual(payload["parameter_prediction_deltas"], [])


if __name__ == "__main__":
    unittest.main()

