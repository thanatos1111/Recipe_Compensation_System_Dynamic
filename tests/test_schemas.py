import unittest

import pandas as pd

from core.schemas import (
    FitArtifacts,
    InstanceCorrectionArtifacts,
    MaterialDataset,
    MaterialModelArtifacts,
    ParameterConfig,
    RecommendationArtifacts,
    SpecConfig,
    TargetInstance,
)


class TestSchemas(unittest.TestCase):
    def test_schema_construction(self) -> None:
        spec = SpecConfig(
            rs_target=10.0,
            rsu_max=1.0,
            use_rs_target_mode=True,
            use_thickness_target_mode=False,
        )

        param_cfg = ParameterConfig(
            name="incident_angle",
            type="continuous",
            min_value=0.0,
            max_value=90.0,
            step=0.5,
            allowed_values=None,
            is_enabled=True,
            is_coupled=False,
            coupling_group=None,
        )
        self.assertEqual(param_cfg.name, "incident_angle")

        ti = TargetInstance(
            target_id="Ta.1",
            material_name="Ta",
            records=pd.DataFrame({"lifetime": [0.0], "rs": [10.0], "rsu": [0.5], "thickness": [100.0]}),
            status="historical",
            fit_artifacts=FitArtifacts(),
            instance_correction_artifacts=InstanceCorrectionArtifacts(),
            warnings=[],
        )
        self.assertEqual(ti.target_id, "Ta.1")

        model_artifacts = MaterialModelArtifacts(
            rs_model=None,
            thickness_model=None,
            rsu_model=None,
            spec_classifier=None,
            feature_schema={"features": []},
            metrics={},
            train_summary={},
            confidence_summary={},
        )

        rec_artifacts = RecommendationArtifacts(
            candidate_table=pd.DataFrame(),
            recommended_recipe={"incident_angle": 10.0},
            predicted_outputs={"rs": 10.0, "thickness": 100.0, "rsu": 0.5},
            score_breakdown={"total": 0.0},
            safe_band={},
            confidence_summary={},
        )

        md = MaterialDataset(
            material_name="Ta",
            source_sheet_name="Ta",
            all_records=pd.DataFrame(),
            target_instances={"Ta.1": ti},
            column_map={"Target ID": "target_id"},
            spec_config=spec,
            material_model_artifacts=model_artifacts,
            recommendation_artifacts=rec_artifacts,
            warnings=[],
        )

        self.assertEqual(md.material_name, "Ta")
        self.assertIn("Ta.1", md.target_instances)


if __name__ == "__main__":
    unittest.main()

