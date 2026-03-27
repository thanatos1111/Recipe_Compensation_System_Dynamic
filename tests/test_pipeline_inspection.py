import unittest

import pandas as pd

from core.pipeline_inspection import build_processing_view_data
from core.schemas import MaterialDataset, TargetInstance


class TestPipelineInspection(unittest.TestCase):
    def test_build_processing_view_data_has_expected_stages(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "target_id": "Ta.1",
                    "lifetime": 1.0,
                    "incident_angle": 10.0,
                    "linear_offset": 0.1,
                    "rotations": 100.0,
                    "ar_flow": 20.0,
                    "o2_flow": 10.0,
                    "rs": 50.0,
                    "thickness": 1000.0,
                    "rsu": 0.8,
                    "in_spec": True,
                    "total_flow": 30.0,
                    "o2_ratio": 0.333,
                },
                {
                    "target_id": "Ta.2",
                    "lifetime": 2.0,
                    "incident_angle": 11.0,
                    "linear_offset": 0.2,
                    "rotations": 110.0,
                    "ar_flow": 21.0,
                    "o2_flow": 9.0,
                    "rs": 52.0,
                    "thickness": 998.0,
                    "rsu": 0.7,
                    "in_spec": False,
                    "total_flow": 30.0,
                    "o2_ratio": 0.3,
                },
            ]
        )
        dataset = MaterialDataset(
            material_name="M1",
            source_sheet_name="M1",
            all_records=df.copy(),
            target_instances={
                "Ta.1": TargetInstance(target_id="Ta.1", material_name="M1", records=df[df["target_id"] == "Ta.1"].copy()),
                "Ta.2": TargetInstance(target_id="Ta.2", material_name="M1", records=df[df["target_id"] == "Ta.2"].copy()),
            },
        )
        out = build_processing_view_data(
            material_dataset=dataset,
            active_target_id="Ta.2",
            config={"feature_config": {}},
        )
        self.assertIn("stages", out)
        self.assertEqual(len(out["stages"]), 8)
        self.assertIn("feature_list", out)
        names = [s["name"] for s in out["stages"]]
        self.assertIn("Raw rows", names)
        self.assertIn("Rows used for training", names)


if __name__ == "__main__":
    unittest.main()
