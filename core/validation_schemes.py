"""
Reusable train/test split generators for material-local, leakage-free benchmarks.

Each function operates on a single material dataframe. Callers must not mix materials.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def make_forward_chaining_splits(
    df: pd.DataFrame,
    *,
    target_id_col: str = "target_id",
    lifetime_col: str = "lifetime",
    min_train_rows: int = 10,
    min_test_rows: int = 3,
    n_splits: int = 3,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """
    Sort by ``target_id`` then ``lifetime``, then yield expanding-window splits.

    Each fold uses strictly earlier rows for training and a contiguous later block for
    testing. Train and test index sets never overlap.
    """
    if df is None or df.empty:
        return
    if target_id_col not in df.columns or lifetime_col not in df.columns:
        return

    work = df.copy()
    work[target_id_col] = work[target_id_col].astype(str)
    work[lifetime_col] = pd.to_numeric(work[lifetime_col], errors="coerce")
    work = work.dropna(subset=[target_id_col, lifetime_col])
    work = work.sort_values([target_id_col, lifetime_col])

    n = len(work)
    if n < min_train_rows + min_test_rows:
        return

    total_test_budget = n - min_train_rows
    max_splits = max(1, total_test_budget // min_test_rows)
    n_folds = min(max(1, n_splits), max_splits)
    if n_folds <= 0:
        return

    leftover = total_test_budget - n_folds * min_test_rows
    sizes = [min_test_rows] * n_folds
    for i in range(leftover):
        sizes[i % n_folds] += 1

    pos = min_train_rows
    for fold in range(n_folds):
        sz = sizes[fold]
        if pos + sz > n:
            break
        train_idx = work.index[:pos]
        test_idx = work.index[pos : pos + sz]
        pos += sz
        yield train_idx, test_idx


def make_leave_one_target_out_splits(
    df: pd.DataFrame,
    *,
    target_id_col: str = "target_id",
    min_train_rows: int = 10,
    min_test_rows: int = 3,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """
    Hold out one ``target_id`` at a time: test = that instance, train = all other rows.
    """
    if df is None or df.empty or target_id_col not in df.columns:
        return

    work = df.copy()
    work[target_id_col] = work[target_id_col].astype(str)
    targets = [str(t) for t in work[target_id_col].unique()]
    for tid in targets:
        test_mask = work[target_id_col] == tid
        train_mask = ~test_mask
        train_idx = work.index[train_mask]
        test_idx = work.index[test_mask]
        if len(train_idx) < min_train_rows or len(test_idx) < min_test_rows:
            continue
        yield train_idx, test_idx


def make_active_target_cutoff_split(
    df: pd.DataFrame,
    *,
    active_target_id: str,
    cutoff_lifetime: float,
    history_target_ids: tuple[str, ...] = (),
    target_id_col: str = "target_id",
    lifetime_col: str = "lifetime",
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """
    Train on optional history targets plus active-target rows with lifetime <= cutoff;
    test on active-target rows with lifetime > cutoff.
    """
    if df is None or df.empty:
        return
    if target_id_col not in df.columns or lifetime_col not in df.columns:
        return

    work = df.copy()
    work[target_id_col] = work[target_id_col].astype(str)
    work[lifetime_col] = pd.to_numeric(work[lifetime_col], errors="coerce")
    work = work.dropna(subset=[target_id_col, lifetime_col])

    active = str(active_target_id)
    history_ids = {str(h) for h in history_target_ids if str(h).strip()}

    history_df = work[work[target_id_col].isin(history_ids)] if history_ids else pd.DataFrame(columns=work.columns)
    active_rows = work[work[target_id_col] == active].copy()
    if active_rows.empty:
        return

    cutoff = float(cutoff_lifetime)
    train_active = active_rows[active_rows[lifetime_col] <= cutoff]
    test_df = active_rows[active_rows[lifetime_col] > cutoff]

    train_df = pd.concat([history_df, train_active], ignore_index=False)
    train_df = train_df.sort_values([target_id_col, lifetime_col])
    test_df = test_df.sort_values(lifetime_col)

    if train_df.empty or test_df.empty:
        return

    yield train_df.index, test_df.index


def split_strategy_to_iter(
    df: pd.DataFrame,
    split_strategy: str,
    extra: dict[str, Any],
) -> Iterator[tuple[str, str, pd.Index, pd.Index]]:
    """
    Map a strategy name to named splits: yields (split_name, split_type, train_idx, test_idx).
    """
    st = (split_strategy or "forward_chaining").strip().lower()

    if st == "forward_chaining":
        fc = {**{"n_splits": 3, "min_train_rows": 10, "min_test_rows": 3}, **extra}
        for i, (tr, te) in enumerate(
            make_forward_chaining_splits(
                df,
                target_id_col=fc.get("target_id_col", "target_id"),
                lifetime_col=fc.get("lifetime_col", "lifetime"),
                min_train_rows=int(fc["min_train_rows"]),
                min_test_rows=int(fc["min_test_rows"]),
                n_splits=int(fc["n_splits"]),
            )
        ):
            yield (f"forward_fold_{i}", "forward_chaining", tr, te)
        return

    if st in {"leave_one_target_out", "loto"}:
        lo = {**{"min_train_rows": 10, "min_test_rows": 3}, **extra}
        for tr, te in make_leave_one_target_out_splits(
            df,
            target_id_col=lo.get("target_id_col", "target_id"),
            min_train_rows=int(lo["min_train_rows"]),
            min_test_rows=int(lo["min_test_rows"]),
        ):
            test_targets = sorted({str(x) for x in df.loc[te, lo.get("target_id_col", "target_id")].unique()})
            name = f"loto_{test_targets[0]}" if test_targets else "loto_unknown"
            yield (name, "leave_one_target_out", tr, te)
        return

    if st in {"active_target_cutoff", "active_cutoff"}:
        ac = {
            **{
                "active_target_id": "",
                "cutoff_lifetime": 0.0,
                "history_target_ids": (),
            },
            **extra,
        }
        aid = str(ac["active_target_id"])
        for tr, te in make_active_target_cutoff_split(
            df,
            active_target_id=aid,
            cutoff_lifetime=float(ac["cutoff_lifetime"]),
            history_target_ids=tuple(ac.get("history_target_ids") or ()),
            target_id_col=ac.get("target_id_col", "target_id"),
            lifetime_col=ac.get("lifetime_col", "lifetime"),
        ):
            yield ("active_target_cutoff", "active_target_cutoff", tr, te)
        return

    raise ValueError(f"Unknown benchmark split_strategy: {split_strategy!r}")
