import unittest

import numpy as np
import pandas as pd

from core.model_registry import (
    ExternalModelDependencyMissingError,
    build_preprocessed_regression_pipeline,
    get_model_availability_summary,
    get_supported_model_specs,
    is_model_available,
)


def _dummy_regression_df() -> tuple[pd.DataFrame, pd.Series]:
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


class TestModelRegistryExternal(unittest.TestCase):
    def test_external_specs_have_metadata(self) -> None:
        specs = get_supported_model_specs()
        for name in ("xgb", "lgbm", "catboost"):
            self.assertIn(name, specs)
            for key in (
                "display_name",
                "family",
                "supports_native_categorical",
                "external_dependency",
                "notes",
            ):
                self.assertIn(key, specs[name])

    def test_availability_summary_contains_external_models(self) -> None:
        summary = get_model_availability_summary()
        for name in ("xgb", "lgbm", "catboost"):
            self.assertIn(name, summary)
            self.assertIsInstance(summary[name]["available"], bool)

    def test_unavailable_external_model_raises_controlled_error(self) -> None:
        # Pick one external family; if it is installed, skip this test.
        model_name = "xgb"
        if is_model_available(model_name):
            self.skipTest("xgboost is installed; controlled-missing test skipped.")

        numeric_features = ["lifetime"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0}}

        with self.assertRaises(ExternalModelDependencyMissingError) as ctx:
            build_preprocessed_regression_pipeline(
                model_name,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                config=config,
                random_state=0,
            )

        self.assertIn("unavailable", str(ctx.exception).lower())

    def test_external_models_can_fit_dummy_dataset_when_available(self) -> None:
        df, y = _dummy_regression_df()
        numeric_features = ["lifetime", "incident_angle"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0}}

        for model_name in ("xgb", "lgbm", "catboost"):
            if not is_model_available(model_name):
                continue
            pipeline = build_preprocessed_regression_pipeline(
                model_name,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                config=config,
                random_state=0,
            )
            pipeline.fit(df[numeric_features + categorical_features], y)


if __name__ == "__main__":
    unittest.main()

