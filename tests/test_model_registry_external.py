import unittest

import json
import numpy as np
import pandas as pd

from pathlib import Path

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

    def test_external_normalizer_step_present_for_lgbm(self) -> None:
        if not is_model_available("lgbm"):
            self.skipTest("lightgbm is not installed.")

        pipeline = build_preprocessed_regression_pipeline(
            "lgbm",
            numeric_features=["lifetime", "incident_angle"],
            categorical_features=["target_id"],
            config={"model_settings": {"random_state": 0}},
            random_state=0,
        )
        step_names = [name for (name, _step) in pipeline.steps]
        self.assertIn("external_matrix_normalizer", step_names)

    def test_external_normalizer_step_present_for_xgb_and_catboost(self) -> None:
        for mn in ("xgb", "catboost"):
            if not is_model_available(mn):
                continue

            pipeline = build_preprocessed_regression_pipeline(
                mn,
                numeric_features=["lifetime", "incident_angle"],
                categorical_features=["target_id"],
                config={"model_settings": {"random_state": 0}},
                random_state=0,
            )
            step_names = [name for (name, _step) in pipeline.steps]
            self.assertIn("external_matrix_normalizer", step_names)

    def test_default_config_has_external_hyperparameter_blocks(self) -> None:
        cfg_path = Path("config/default_config.json")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        ms = cfg.get("model_settings") or {}
        for name in ("xgb", "lgbm", "catboost"):
            self.assertIn(name, ms)

        # Minimal required blocks per Prompt 7.1.
        self.assertIn("n_estimators", ms["xgb"])
        self.assertIn("learning_rate", ms["xgb"])
        self.assertIn("n_estimators", ms["lgbm"])
        self.assertIn("learning_rate", ms["lgbm"])
        self.assertIn("iterations", ms["catboost"])
        self.assertIn("learning_rate", ms["catboost"])

    def test_lgbm_warning_fix_feature_names(self) -> None:
        if not is_model_available("lgbm"):
            self.skipTest("lightgbm is not installed.")

        import warnings

        df, y = _dummy_regression_df()
        numeric_features = ["lifetime", "incident_angle"]
        categorical_features = ["target_id"]
        config = {"model_settings": {"random_state": 0}}

        pipeline = build_preprocessed_regression_pipeline(
            "lgbm",
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
            random_state=0,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(df[numeric_features + categorical_features], y)
            _ = pipeline.predict(df[numeric_features + categorical_features])

        # LightGBM warning we want to avoid.
        needle = "does not have valid feature names"
        bad = [w for w in caught if needle in str(w.message).lower() and "lgbmregressor" in str(w.message).lower()]
        self.assertFalse(bad, f"Unexpected LightGBM feature-name warning(s): {[str(w.message) for w in bad]}")

    def test_availability_summary_expected_fields_for_external_models(self) -> None:
        summary = get_model_availability_summary()
        for mn in ("xgb", "lgbm", "catboost"):
            self.assertIn(mn, summary)
            meta = summary[mn]
            for key in ("available", "external_dependency", "missing_error"):
                self.assertIn(key, meta)
            self.assertIsInstance(meta["available"], bool)


if __name__ == "__main__":
    unittest.main()

