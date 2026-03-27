import unittest

import numpy as np
import pandas as pd

from core.schemas import MaterialDataset, SpecConfig, TargetInstance
from core.update_pipeline import append_runs_and_refresh


def _make_df(target_id: str, n: int, *, start_lifetime: float = 0.0, lifetime_step: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for i in range(n):
        lifetime = start_lifetime + i * lifetime_step
        incident_angle = 10.0 + 0.05 * i
        linear_offset = 0.1 + 0.01 * i
        rotations = 100.0 + i
        rpm = 60.0 + (i % 3) * 2.0
        power = 1000.0 + (i % 4) * 50.0
        ar_flow = 20.0
        o2_flow = 10.0
        total_flow = ar_flow + o2_flow
        o2_ratio = o2_flow / total_flow

        # Outputs with a deterministic trend.
        rs = 5000.0 + 0.2 * i + (0.5 if target_id.endswith("2") else 0.0) + float(rng.normal(0, 10.0))
        thickness = 1000.0 + 0.1 * i + (2.0 if target_id.endswith("2") else 0.0) + float(rng.normal(0, 0.5))
        rsu = 3.0 + 0.01 * i + float(rng.normal(0, 0.02))

        rows.append(
            {
                "target_id": target_id,
                "material_name": target_id.split(".")[0],
                "lot_id": "L1",
                "date": pd.Timestamp("2024-01-01"),
                "wafer_id": "W1",
                "lifetime": float(lifetime),
                "incident_angle": float(incident_angle),
                "linear_offset": float(linear_offset),
                "rotations": float(rotations),
                "rpm": float(rpm),
                "power": float(power),
                "ar_flow": float(ar_flow),
                "o2_flow": float(o2_flow),
                "thickness": float(thickness),
                "rs": float(rs),
                "rsu": float(rsu),
            }
        )
    return pd.DataFrame(rows)


class TestUpdatePipeline(unittest.TestCase):
    def test_append_runs_and_refresh_artifacts(self) -> None:
        df1 = _make_df("Ta.1", 25, start_lifetime=0.0)
        df2 = _make_df("Ta.2", 25, start_lifetime=0.0)

        spec = SpecConfig(
            rs_min=0.0,
            rs_max=1e9,
            use_rs_target_mode=False,
            thickness_min=0.0,
            thickness_max=1e9,
            use_thickness_target_mode=False,
            rsu_max=10.0,
            use_rsu_spec=True,
            use_rs_spec=True,
            use_thickness_spec=True,
        )

        # Material dataset with two target instances.
        ds = MaterialDataset(
            material_name="Ta",
            source_sheet_name="Ta",
            all_records=pd.concat([df1, df2], ignore_index=True),
            target_instances={
                "Ta.1": TargetInstance(target_id="Ta.1", material_name="Ta", records=df1.copy()),
                "Ta.2": TargetInstance(target_id="Ta.2", material_name="Ta", records=df2.copy()),
            },
        )
        ds.spec_config = spec

        train_config = {
            "feature_config": {"numeric_features": [], "categorical_features": ["target_id"]},
            "model_settings": {"random_state": 0, "test_fraction": 0.2},
        }
        # Ensure feature_config uses defaults by letting train_material_models fall back.
        # For simplicity, pass empty; build_feature_matrix will fill missing with NA but schema expects features.
        # Our test mostly checks that refresh completes.
        new_rows = _make_df("Ta.1", 3, start_lifetime=25.0, lifetime_step=1.0)
        ds = append_runs_and_refresh(
            ds,
            "Ta.1",
            new_rows,
            train_config=train_config,
            spec_config=spec,
        )

        self.assertEqual(len(ds.target_instances["Ta.1"].records), 28)
        self.assertIsNotNone(ds.material_model_artifacts)
        self.assertIsNotNone(ds.target_instances["Ta.1"].instance_correction_artifacts)

        # Derived columns should exist.
        self.assertIn("lifetime_used", ds.all_records.columns)
        self.assertIn("lifetime_end", ds.all_records.columns)
        self.assertIn("in_spec", ds.all_records.columns)


if __name__ == "__main__":
    unittest.main()

