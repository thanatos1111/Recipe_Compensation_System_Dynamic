import unittest

import pandas as pd

from core.model_explanations import build_training_explanation
from core.schemas import MaterialModelArtifacts


class TestModelExplanations(unittest.TestCase):
    def test_build_training_explanation_with_sparse_data_and_fallback(self) -> None:
        df = pd.DataFrame(
            [
                {"target_id": "T1", "lifetime": 1.0, "in_spec": True, "rs": 10.0},
                {"target_id": "T1", "lifetime": 2.0, "in_spec": False, "rs": 10.2},
                {"target_id": "T2", "lifetime": 3.0, "in_spec": True, "rs": 10.3},
            ]
        )
        artifacts = MaterialModelArtifacts(
            rs_model=None,
            thickness_model=None,
            rsu_model=None,
            metrics={},
            confidence_summary={"level": "Low", "notes": ["very_few_rows"]},
            train_summary={"row_count": 3, "in_spec_count": 2, "target_instance_count": 2},
        )

        out = build_training_explanation(
            train_df=df,
            artifacts=artifacts,
            active_target_id="T1",
            feature_schema={"numeric": ["lifetime"], "categorical": ["target_id"]},
        )

        self.assertIn("Sparse-data condition", " ".join(out.warnings))
        self.assertIn("Fallback mode active", " ".join(out.warnings))
        label_map = {item.label: item for item in out.items}
        self.assertEqual(label_map["In-spec / out-of-spec count"].value, "2 in-spec / 1 out-of-spec")
        self.assertIn("unavailable", label_map["Model type"].value)

    def test_build_training_explanation_extrapolation_warning(self) -> None:
        df = pd.DataFrame(
            [
                {"target_id": "A", "lifetime": 10.0, "in_spec": True},
                {"target_id": "B", "lifetime": 20.0, "in_spec": True},
                {"target_id": "B", "lifetime": 25.0, "in_spec": True},
            ]
        )
        out = build_training_explanation(
            train_df=df,
            artifacts=None,
            active_target_id="B",
            feature_schema={"numeric": [], "categorical": []},
        )
        # Active target B extends beyond lower bound of full train range in this dataset split logic.
        self.assertTrue(any("Extrapolation warning" in warning for warning in out.warnings))


if __name__ == "__main__":
    unittest.main()

