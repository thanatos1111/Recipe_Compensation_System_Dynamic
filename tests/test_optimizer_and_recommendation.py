import unittest

import numpy as np
import pandas as pd

from core.candidate_generator import generate_candidates
from core.constraints import enforce_coupling_rules
from core.optimizer import rank_candidates, select_best_candidate
from core.safe_band import estimate_multi_parameter_bands, estimate_parameter_band
from core.schemas import SpecConfig


class DummyModel:
    def __init__(self, col: str) -> None:
        self.col = col

    def predict(self, X: pd.DataFrame):
        # Return as numpy array for sklearn-like compatibility.
        return X[self.col].to_numpy(dtype=float)


class TestOptimizerAndRecommendation(unittest.TestCase):
    def test_coupling_total_flow_enforced(self) -> None:
        cand = {"o2_flow": 12.0, "ar_flow": 20.0}
        out = enforce_coupling_rules(cand, {"coupling_group": "total_flow", "total_flow_value": 30.0})
        self.assertAlmostEqual(out["ar_flow"], 18.0)

    def test_candidate_generation_respects_total_flow(self) -> None:
        reference = {
            "incident_angle": 10.0,
            "linear_offset": 0.1,
            "rotations": 100.0,
            "ar_flow": 20.0,
            "o2_flow": 10.0,
            "total_flow": 30.0,
        }

        parameter_config = {
            "incident_angle": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": False,
                "coupling_group": None,
            },
            "linear_offset": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 10.0,
                "step": 0.1,
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
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": True,
                "coupling_group": "total_flow",
            },
            "o2_flow": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": True,
                "coupling_group": "total_flow",
            },
        }

        candidates = generate_candidates(reference, parameter_config, {"radius_steps": 1, "max_candidates": 1000})
        self.assertFalse(candidates.empty)
        for _, row in candidates.iterrows():
            self.assertAlmostEqual(float(row["ar_flow"]) + float(row["o2_flow"]), 30.0, places=6)

    def test_score_ranking_prefers_target_rs(self) -> None:
        # Spec: RS target mode around 12 +/- tol 1.0
        spec = SpecConfig(
            rs_target=12.0,
            rs_tol=1.0,
            use_rs_target_mode=True,
            rsu_max=0.0,
            use_rsu_spec=False,  # isolate rs behavior
            use_rs_spec=True,
            use_thickness_spec=False,
            use_thickness_target_mode=False,
        )

        reference_recipe = {
            "incident_angle": 10.0,
            "linear_offset": 0.0,
            "rotations": 100.0,
            "ar_flow": 20.0,
            "o2_flow": 10.0,
            "total_flow": 30.0,
        }

        parameter_config = {
            "incident_angle": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": False,
                "coupling_group": None,
            },
            "ar_flow": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": False,
                "is_coupled": False,
                "coupling_group": None,
            },
            "o2_flow": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 100.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": False,
                "is_coupled": False,
                "coupling_group": None,
            },
            "linear_offset": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.1,
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
        }

        candidates = generate_candidates(reference_recipe, parameter_config, {"radius_steps": 2, "max_candidates": 200})

        feature_config = {
            "numeric_features": [
                "lifetime",
                "lifetime_used",
                "lifetime_end",
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
        }

        # Dummy models:
        # rs_pred = incident_angle
        # thickness_pred and rsu_pred are constant but disabled in spec scoring.
        model_bundle = {
            "rs_model": DummyModel("incident_angle"),
            "thickness_model": DummyModel("linear_offset"),
            "rsu_model": DummyModel("o2_flow"),
            "feature_config": feature_config,
            "instance_correction": None,
        }

        context = {
            "spec_config": spec,
            "weights": {"w_rs": 1.0, "w_thickness": 0.0, "w_rsu": 0.0, "w_move": 0.0, "w_margin": 0.0},
            "reference_recipe": reference_recipe,
            "active_target_id": "Ta.1",
            "lifetime": 0.0,
            "rpm": 60.0,
            "power": 1000.0,
            "feature_config": feature_config,
            "instance_correction": None,
            "parameter_config": parameter_config,
        }

        ranked = rank_candidates(candidates, model_bundle, context)
        self.assertFalse(ranked.empty)
        best = select_best_candidate(ranked)
        # Best should have rs_pred (incident_angle) closest to 12.
        self.assertAlmostEqual(float(best["incident_angle"]), 12.0, places=6)

    def test_safe_band_detects_feasible_interval(self) -> None:
        spec = SpecConfig(
            rs_target=10.0,
            rs_tol=1.0,
            use_rs_target_mode=True,
            use_rs_spec=True,
            use_thickness_spec=False,
            use_rsu_spec=False,
            rsu_max=0.0,
            use_thickness_target_mode=False,
        )

        feature_config = {
            "numeric_features": [
                "lifetime",
                "lifetime_used",
                "lifetime_end",
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
        }

        model_bundle = {
            "rs_model": DummyModel("incident_angle"),
            "thickness_model": DummyModel("linear_offset"),
            "rsu_model": DummyModel("o2_flow"),
            "feature_config": feature_config,
            "instance_correction": None,
        }

        center = {
            "incident_angle": 10.0,
            "linear_offset": 0.0,
            "rotations": 100.0,
            "ar_flow": 20.0,
            "o2_flow": 10.0,
            "target_id": "Ta.1",
            "lifetime": 0.0,
            "rpm": 60.0,
            "power": 1000.0,
        }

        band = estimate_parameter_band(center, model_bundle, parameter_name="incident_angle", spec_config=spec, step=0.5, max_steps=4)
        # With target 10 +/- 1.0, feasible incident_angle should be [9.0, 11.0].
        self.assertAlmostEqual(band["lower"], 9.0, places=6)
        self.assertAlmostEqual(band["upper"], 11.0, places=6)

    def test_minimum_step_defaults_applied_when_step_is_zero(self) -> None:
        # incident_angle step is 0 -> should fall back to minimum step (0.01).
        reference = {
            "incident_angle": 10.0,
        }
        parameter_config = {
            "incident_angle": {
                "type": "continuous",
                "min_value": 0.0,
                "max_value": 1000.0,
                "step": 0.0,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": False,
                "coupling_group": None,
            },
            "rotations": {
                "type": "integer",
                "min_value": 0.0,
                "max_value": 100000.0,
                "step": 1.0,
                "allowed_values": None,
                "is_enabled": False,
                "is_coupled": False,
                "coupling_group": None,
            },
        }

        candidates = generate_candidates(reference, parameter_config, {"radius_steps": 1, "max_candidates": 50})
        self.assertFalse(candidates.empty)

        step = 0.01
        for _, row in candidates.iterrows():
            v = float(row["incident_angle"])
            q = round(v / step) * step
            self.assertAlmostEqual(v, q, places=6)

    def test_multi_parameter_safe_band_reports_expected_ranges(self) -> None:
        spec = SpecConfig(
            rs_target=10.0,
            rs_tol=1.0,
            use_rs_target_mode=True,
            use_rs_spec=True,
            use_thickness_spec=False,
            use_rsu_spec=False,
            rsu_max=0.0,
            use_thickness_target_mode=False,
        )

        feature_config = {
            "numeric_features": [
                "lifetime",
                "lifetime_used",
                "lifetime_end",
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
        }
        model_bundle = {
            "rs_model": DummyModel("incident_angle"),
            "thickness_model": DummyModel("linear_offset"),
            "rsu_model": DummyModel("o2_flow"),
            "feature_config": feature_config,
            "instance_correction": None,
        }
        center = {
            "incident_angle": 10.0,
            "linear_offset": 0.0,
            "rotations": 100.0,
            "ar_flow": 20.0,
            "o2_flow": 10.0,
            "target_id": "Ta.1",
            "lifetime": 0.0,
            "rpm": 60.0,
            "power": 1000.0,
        }

        out = estimate_multi_parameter_bands(
            center,
            model_bundle,
            parameter_names=["incident_angle", "linear_offset"],
            spec_config=spec,
            parameter_steps={"incident_angle": 0.5, "linear_offset": 0.1},
            candidate_radius_steps=2,
            safe_band_max_steps=4,
            clip_to_candidate_range=False,
        )

        ia = out["incident_angle"]
        self.assertEqual(ia["candidate_search_range"], [9.0, 11.0])
        self.assertEqual(ia["safe_band_search_range"], [8.0, 12.0])
        self.assertEqual(ia["feasible_band"], [9.0, 11.0])

        lo = out["linear_offset"]
        # linear_offset does not affect RS in this dummy setup, so feasible
        # band spans full safe-band search range.
        self.assertEqual(lo["candidate_search_range"], [-0.2, 0.2])
        self.assertEqual(lo["safe_band_search_range"], [-0.4, 0.4])
        self.assertEqual(lo["feasible_band"], [-0.4, 0.4])

    def test_safe_band_can_be_clipped_to_candidate_neighborhood(self) -> None:
        spec = SpecConfig(
            rs_target=10.0,
            rs_tol=1.0,
            use_rs_target_mode=True,
            use_rs_spec=True,
            use_thickness_spec=False,
            use_rsu_spec=False,
            rsu_max=0.0,
            use_thickness_target_mode=False,
        )
        feature_config = {
            "numeric_features": [
                "lifetime",
                "lifetime_used",
                "lifetime_end",
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
        }
        model_bundle = {
            "rs_model": DummyModel("incident_angle"),
            "thickness_model": DummyModel("linear_offset"),
            "rsu_model": DummyModel("o2_flow"),
            "feature_config": feature_config,
            "instance_correction": None,
        }
        center = {
            "incident_angle": 10.0,
            "linear_offset": 0.0,
            "rotations": 100.0,
            "ar_flow": 20.0,
            "o2_flow": 10.0,
            "target_id": "Ta.1",
            "lifetime": 0.0,
            "rpm": 60.0,
            "power": 1000.0,
        }

        unclipped = estimate_multi_parameter_bands(
            center,
            model_bundle,
            parameter_names=["linear_offset"],
            spec_config=spec,
            parameter_steps={"linear_offset": 0.1},
            candidate_radius_steps=1,
            safe_band_max_steps=4,
            clip_to_candidate_range=False,
        )["linear_offset"]
        clipped = estimate_multi_parameter_bands(
            center,
            model_bundle,
            parameter_names=["linear_offset"],
            spec_config=spec,
            parameter_steps={"linear_offset": 0.1},
            candidate_radius_steps=1,
            safe_band_max_steps=4,
            clip_to_candidate_range=True,
        )["linear_offset"]

        self.assertEqual(unclipped["candidate_search_range"], [-0.1, 0.1])
        self.assertEqual(unclipped["feasible_band"], [-0.4, 0.4])
        self.assertEqual(clipped["candidate_search_range"], [-0.1, 0.1])
        self.assertEqual(clipped["feasible_band"], [-0.1, 0.1])


if __name__ == "__main__":
    unittest.main()

