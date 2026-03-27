"""
Export utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.schemas import RecommendationArtifacts


def export_recommendation_table(
    artifacts: RecommendationArtifacts,
    out_path: str,
) -> None:
    """
    Export recommendation candidates to CSV.

    If `out_path` points to a directory (or has no suffix), writes:
    - `<out_path>/recommendation_candidates.csv`
    Otherwise writes directly to `out_path`.
    """
    if artifacts is None:
        raise ValueError("artifacts must not be None")

    path = Path(out_path)
    candidate_df = artifacts.candidate_table
    if candidate_df is None:
        candidate_df = pd.DataFrame()

    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate_df.to_csv(path, index=False)
        return

    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "recommendation_candidates.csv"
    candidate_df.to_csv(csv_path, index=False)


def export_plots(out_path: str, *, plots: dict[str, Any]) -> None:
    """
    Export plot objects to image files.

    Supported values in `plots`:
    - matplotlib Figure-like objects (must have `savefig`)
    - raw image bytes/bytearray
    """
    export_dir = Path(out_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    for key, obj in (plots or {}).items():
        name = str(key).strip() or "plot"
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)

        if hasattr(obj, "savefig"):
            file_path = export_dir / f"{safe_name}.png"
            obj.savefig(file_path)
            continue

        if isinstance(obj, (bytes, bytearray)):
            file_path = export_dir / f"{safe_name}.png"
            file_path.write_bytes(bytes(obj))
            continue

        raise TypeError(f"Unsupported plot object for '{key}': {type(obj).__name__}")

