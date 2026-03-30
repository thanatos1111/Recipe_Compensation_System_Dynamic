"""Tests for material-local benchmark split generators."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.validation_schemes import (
    make_active_target_cutoff_split,
    make_forward_chaining_splits,
    make_leave_one_target_out_splits,
)


def _synth_df(*, n_per_target: int = 20, targets: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for tid in targets:
        for i in range(n_per_target):
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
                    "rs": float(rng.normal(50, 0.1)),
                    "thickness": 1000.0,
                    "rsu": 0.5,
                }
            )
    return pd.DataFrame(rows)


class TestValidationSchemes(unittest.TestCase):
    def test_forward_chaining_no_train_test_overlap(self) -> None:
        df = _synth_df(n_per_target=25, targets=("T1", "T2"))
        splits = list(
            make_forward_chaining_splits(
                df,
                min_train_rows=10,
                min_test_rows=3,
                n_splits=3,
            )
        )
        self.assertGreater(len(splits), 0)
        for tr, te in splits:
            self.assertEqual(len(tr.intersection(te)), 0)
            self.assertGreater(len(tr), 0)
            self.assertGreaterEqual(len(te), 3)

    def test_leave_one_target_out_isolates_held_out_target(self) -> None:
        df = _synth_df(n_per_target=15, targets=("X", "Y", "Z"))
        for tr, te in make_leave_one_target_out_splits(df, min_train_rows=10, min_test_rows=3):
            train_targets = set(df.loc[tr, "target_id"].astype(str))
            test_targets = set(df.loc[te, "target_id"].astype(str))
            self.assertEqual(len(test_targets), 1)
            held = test_targets.pop()
            self.assertNotIn(held, train_targets)
            self.assertEqual(train_targets | {held}, {"X", "Y", "Z"})

    def test_active_target_cutoff_train_before_test_in_time(self) -> None:
        df = _synth_df(n_per_target=30, targets=("hist", "act"))
        # Only use "act" as active; "hist" is optional history.
        sub = df[df["target_id"].isin(["hist", "act"])].copy()
        out = list(
            make_active_target_cutoff_split(
                sub,
                active_target_id="act",
                cutoff_lifetime=10.0,
                history_target_ids=("hist",),
            )
        )
        self.assertEqual(len(out), 1)
        tr, te = out[0]
        act_train = sub.loc[tr]
        act_test = sub.loc[te]
        self.assertTrue((act_train[act_train["target_id"] == "act"]["lifetime"] <= 10.0).all())
        self.assertTrue((act_test["lifetime"] > 10.0).all())
        self.assertEqual(len(tr.intersection(te)), 0)


if __name__ == "__main__":
    unittest.main()
