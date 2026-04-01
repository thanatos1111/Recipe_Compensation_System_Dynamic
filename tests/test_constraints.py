from __future__ import annotations

import unittest

from core.constraints import enforce_discrete_step, validate_candidate_parameters


class TestConstraints(unittest.TestCase):
    def test_dependent_constraints_enforced_and_clamped(self) -> None:
        cfg = {
            "linear_offset": {"type": "continuous", "min_value": -190.0, "max_value": 190.0, "step": 1.0, "is_enabled": True},
            "incident_angle": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 90.0,
                "step": 1.0,
                "is_enabled": True,
                "dependent_constraints": [
                    {
                        "driver_parameter": "linear_offset",
                        "rules": [
                            {"driver_min": -190.0, "driver_max": -175.0, "min_value": 0.0, "max_value": 46.0},
                            {"driver_min": -175.0, "driver_max": 15.0, "min_value": 0.0, "max_value": 50.0},
                        ],
                    }
                ],
            },
        }
        cand = {"linear_offset": -180.0, "incident_angle": 80.0}
        viol = validate_candidate_parameters(cand, cfg)
        self.assertTrue(any(v == "incident_angle_above_max" for v in viol))

        snapped = enforce_discrete_step(cand, cfg)
        self.assertLessEqual(float(snapped["incident_angle"]), 50.0)
        self.assertEqual(validate_candidate_parameters(snapped, cfg), [])


if __name__ == "__main__":
    unittest.main()

