"""Leakage-free benchmark evaluation tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.response_models import evaluate_models_with_splits, fit_model_for_target, train_material_models
from core.schemas import BenchmarkFoldResult, BenchmarkSummary, SpecConfig


def _make_material_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for tid in ("P", "Q"):
        for i in range(n // 2):
            rows.append(
                {
                    "target_id": tid,
                    "lifetime": float(i),
                    "incident_angle": 10.0,
                    "linear_offset": 0.5,
                    "rotations": 100.0,
                    "ar_flow": 10.0,
                    "o2_flow": 5.0,
                    "o2_ratio": 5.0 / 15.0,
                    "total_flow": 15.0,
                    "rs": 50.0 + 0.1 * i + float(rng.normal(0, 0.05)),
                    "thickness": 1000.0 + float(rng.normal(0, 0.1)),
                    "rsu": 0.5,
                    "in_spec": True,
                }
            )
    return pd.DataFrame(rows)


class TestLeakageFreeEvaluation(unittest.TestCase):
    def test_benchmark_summary_fold_and_aggregate_metrics(self) -> None:
        df = _make_material_df(40)
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "split_strategy": "forward_chaining",
                "forward_chaining": {"n_splits": 2, "min_train_rows": 10, "min_test_rows": 4},
            },
        }
        out = evaluate_models_with_splits(
            df,
            config=cfg,
            split_strategy="forward_chaining",
            spec_config=SpecConfig(),
        )
        self.assertIn("folds", out)
        self.assertIn("summary", out)
        self.assertGreaterEqual(out.get("n_folds", 0), 1)
        self.assertIsNotNone(out.get("rs_mae"))

    def test_train_material_models_benchmark_separate_from_deployment(self) -> None:
        df = _make_material_df(40)
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "split_strategy": "forward_chaining",
                "forward_chaining": {"n_splits": 2, "min_train_rows": 10, "min_test_rows": 4},
            },
        }
        artifacts = train_material_models(df, config=cfg, spec_config=SpecConfig())
        self.assertIsNotNone(artifacts.rs_model)
        self.assertIn("folds", artifacts.metrics)
        self.assertIn("benchmark_split_type", artifacts.metrics)

    def test_evaluator_fits_only_on_train_rows(self) -> None:
        df = _make_material_df(60)
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {
                "forward_chaining": {"n_splits": 2, "min_train_rows": 12, "min_test_rows": 5},
            },
        }
        seen_train_lens: list[int] = []

        def wrapped_fit(
            df_train: pd.DataFrame,
            target_col: str,
            config: dict,
            *,
            feature_matrix: object = None,
        ) -> object:
            seen_train_lens.append(len(df_train))
            return fit_model_for_target(df_train, target_col, config, feature_matrix=feature_matrix)

        with patch("core.response_models.fit_model_for_target", side_effect=wrapped_fit):
            evaluate_models_with_splits(df, config=cfg, split_strategy="forward_chaining", spec_config=None)

        self.assertTrue(seen_train_lens)
        full = len(df)
        self.assertTrue(all(n < full for n in seen_train_lens))

    def test_no_cross_material_mixing_is_callers_responsibility(self) -> None:
        """Splits only see the dataframe passed in; two materials stay isolated if given separate dfs."""
        df_a = _make_material_df(20)
        df_b = _make_material_df(20).copy()
        df_b["target_id"] = "Z"
        cfg = {
            "feature_config": {},
            "model_settings": {"random_state": 0},
            "benchmark_settings": {"forward_chaining": {"n_splits": 1, "min_train_rows": 8, "min_test_rows": 3}},
        }
        m_a = evaluate_models_with_splits(df_a, config=cfg, split_strategy="forward_chaining")
        m_b = evaluate_models_with_splits(df_b, config=cfg, split_strategy="forward_chaining")
        self.assertNotEqual(m_a.get("folds"), m_b.get("folds"))

    def test_benchmark_dataclasses_import(self) -> None:
        f = BenchmarkFoldResult(
            split_name="x",
            split_type="forward_chaining",
            train_row_count=1,
            test_row_count=1,
            train_target_ids=["a"],
            test_target_ids=["b"],
        )
        s = BenchmarkSummary(split_type="forward_chaining", folds=[f])
        self.assertEqual(s.folds[0].split_name, "x")


if __name__ == "__main__":
    unittest.main()
