"""
Core typed schema definitions for the recipe compensation system.

Milestone 1 only: data containers + type hints (no business logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class SpecConfig:
    rs_target: Optional[float] = None
    rs_min: Optional[float] = None
    rs_max: Optional[float] = None
    rs_tol: Optional[float] = None

    thickness_target: Optional[float] = None
    thickness_min: Optional[float] = None
    thickness_max: Optional[float] = None
    thickness_tol: Optional[float] = None

    rsu_max: float = 0.0
    use_rs_target_mode: bool = False
    use_thickness_target_mode: bool = False

    # Enable/disable individual spec components in in_spec labeling.
    # This allows operators to turn off parts of the filter without
    # editing numeric thresholds.
    use_rs_spec: bool = True
    use_thickness_spec: bool = True
    use_rsu_spec: bool = True


@dataclass(frozen=True)
class ParameterConfig:
    # Recipe parameter metadata used by validation + candidate generation.
    name: str
    type: str  # "continuous", "integer", "categorical", "boolean"
    min_value: float
    max_value: float
    step: float
    allowed_values: Optional[list[Any]] = None
    is_enabled: bool = True
    is_coupled: bool = False
    coupling_group: Optional[str] = None


@dataclass
class BenchmarkFoldResult:
    """Per-fold metrics from a leakage-free benchmark run (one material)."""

    split_name: str
    split_type: str
    train_row_count: int
    test_row_count: int
    train_target_ids: list[str]
    test_target_ids: list[str]
    rs_mae: Optional[float] = None
    rs_rmse: Optional[float] = None
    thickness_mae: Optional[float] = None
    thickness_rmse: Optional[float] = None
    rsu_mae: Optional[float] = None
    rsu_rmse: Optional[float] = None
    spec_pass_accuracy: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    """Aggregate statistics over benchmark folds."""

    split_type: str
    folds: list[BenchmarkFoldResult]
    rs_mae_mean: Optional[float] = None
    rs_mae_std: Optional[float] = None
    rs_rmse_mean: Optional[float] = None
    rs_rmse_std: Optional[float] = None
    thickness_mae_mean: Optional[float] = None
    thickness_mae_std: Optional[float] = None
    thickness_rmse_mean: Optional[float] = None
    thickness_rmse_std: Optional[float] = None
    rsu_mae_mean: Optional[float] = None
    rsu_mae_std: Optional[float] = None
    rsu_rmse_mean: Optional[float] = None
    rsu_rmse_std: Optional[float] = None
    spec_pass_accuracy_mean: Optional[float] = None
    spec_pass_accuracy_std: Optional[float] = None
    aggregate_warnings: list[str] = field(default_factory=list)


@dataclass
class MaterialModelArtifacts:
    rs_model: Any = None
    thickness_model: Any = None
    rsu_model: Any = None

    spec_classifier: Any = None
    feature_schema: dict[str, Any] = field(default_factory=dict)

    metrics: dict[str, Any] = field(default_factory=dict)
    train_summary: dict[str, Any] = field(default_factory=dict)
    confidence_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceCorrectionArtifacts:
    bias_terms: dict[str, Any] = field(default_factory=dict)
    recent_residual_summary: dict[str, Any] = field(default_factory=dict)
    instance_confidence: str = "unknown"
    drift_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitArtifacts:
    parameter_trend_fits: dict[str, Any] = field(default_factory=dict)
    lifetime_bins: list[tuple[float, float]] = field(default_factory=list)
    in_spec_region_maps: dict[str, Any] = field(default_factory=dict)
    stepwise_reference_table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


@dataclass
class RecommendationArtifacts:
    candidate_table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    recommended_recipe: dict[str, Any] = field(default_factory=dict)
    predicted_outputs: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    safe_band: dict[str, Any] = field(default_factory=dict)
    confidence_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetInstance:
    target_id: str
    material_name: str
    records: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Suggested statuses from the technical spec:
    # "historical" | "active" | "retired"
    status: str = "historical"

    fit_artifacts: Optional[FitArtifacts] = None
    instance_correction_artifacts: Optional[InstanceCorrectionArtifacts] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class MaterialDataset:
    # Represents one material sheet + grouped target instances.
    material_name: str
    source_sheet_name: str

    all_records: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    target_instances: dict[str, TargetInstance] = field(default_factory=dict)

    column_map: dict[str, Any] = field(default_factory=dict)
    spec_config: SpecConfig = field(default_factory=SpecConfig)

    material_model_artifacts: MaterialModelArtifacts = field(default_factory=MaterialModelArtifacts)
    recommendation_artifacts: Optional[RecommendationArtifacts] = None

    warnings: list[str] = field(default_factory=list)

