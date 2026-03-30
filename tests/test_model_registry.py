import unittest

import numpy as np
import pandas as pd

from core.model_registry import (
    build_preprocessed_regression_pipeline,
    get_supported_model_specs,
)


class TestModelRegistry(unittest.TestCase):
    def _dummy_regression_df(self) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "lifetime": np.linspace(0, 10, 30),
                "incident_angle": np.linspace(5, 15, 30),
                "target_id": ["T1"] * 15 + ["T2"] * 15,
            }
        )
        # Simple linear target with noise.
        y = 0.5 * df["lifetime"].to_numpy() + 0.1 * df["incident_angle"].to_numpy() + rng.normal(0, 0.01, size=len(df))
        return df, pd.Series(y, name="y")

    def test_all_declared_models_build(self) -> None:
        supported = get_supported_model_specs()
        self.assertGreater(len(supported), 0)

        numeric_features = ["lifetime", "incident_angle"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0}}

        df, _y = self._dummy_regression_df()
        for model_name in supported.keys():
            pipeline = build_preprocessed_regression_pipeline(
                model_name,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                config=config,
                random_state=0,
            )
            self.assertIsNotNone(pipeline)

            # Fit a tiny model to ensure preprocessing + estimator wiring works.
            pipeline.fit(df[numeric_features + categorical_features], df["lifetime"])

    def test_pipeline_can_fit_dummy_dataframe(self) -> None:
        numeric_features = ["lifetime", "incident_angle"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0, "mlp": {"max_iter": 50}}}

        df, y = self._dummy_regression_df()
        pipeline = build_preprocessed_regression_pipeline(
            "mlp",
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
            random_state=0,
        )
        pipeline.fit(df[numeric_features + categorical_features], y)

    def test_unknown_model_raises_clear_error(self) -> None:
        numeric_features = ["lifetime"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0}}

        with self.assertRaises(ValueError) as ctx:
            build_preprocessed_regression_pipeline(
                "not_a_real_model",
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                config=config,
                random_state=0,
            )
        self.assertIn("Unknown regression model name", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

