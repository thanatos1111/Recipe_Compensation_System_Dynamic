"""
Model registry for recipe parameter regression.

This module centralizes sklearn estimator + preprocessing construction so that
future ML methods can be benchmarked via a consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

import numpy as np
import pandas as pd


_SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "linear",
    "ridge",
    "elastic_net",
    "rf",
    "gbr",
    "hgbt",
    "mlp",
    # External gradient boosting regressors (optional dependencies).
    "xgb",
    "lgbm",
    "catboost",
)


_EXTERNAL_MODEL_DEPENDENCIES: dict[str, str] = {
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "catboost": "catboost",
}


_XGBRegressor = None
_LGBMRegressor = None
_CatBoostRegressor = None

try:
    from xgboost import XGBRegressor as _XGBRegressor  # type: ignore[import-not-found]
except Exception:
    _XGBRegressor = None

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor  # type: ignore[import-not-found]
except Exception:
    _LGBMRegressor = None

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor  # type: ignore[import-not-found]
except Exception:
    _CatBoostRegressor = None


@dataclass(frozen=True)
class ExternalModelDependencyMissingError(RuntimeError):
    """Raised when an external-model family is requested but not installed."""

    model_name: str
    dependency: str

    def __str__(self) -> str:  # pragma: no cover
        return self.user_message

    @property
    def user_message(self) -> str:
        return (
            f"Model {self.model_name!r} unavailable: missing external dependency {self.dependency!r}. "
            f"Install it to enable this model family."
        )


def is_model_available(model_name: str) -> bool:
    """Return whether a supported model can currently be constructed."""
    mn = (model_name or "").strip().lower()
    if mn not in _SUPPORTED_MODEL_NAMES:
        return False
    if mn == "xgb":
        return _XGBRegressor is not None
    if mn == "lgbm":
        return _LGBMRegressor is not None
    if mn == "catboost":
        return _CatBoostRegressor is not None
    return True


def get_model_availability_summary() -> dict[str, dict[str, Any]]:
    """Return a UI-friendly availability summary for supported model names."""
    out: dict[str, dict[str, Any]] = {}
    for name in _SUPPORTED_MODEL_NAMES:
        if name in _EXTERNAL_MODEL_DEPENDENCIES:
            dep = _EXTERNAL_MODEL_DEPENDENCIES[name]
            available = is_model_available(name)
            out[name] = {
                "available": available,
                "external_dependency": dep,
                "missing_error": None if available else f"pip install {dep}",
            }
        else:
            out[name] = {"available": True, "external_dependency": None, "missing_error": None}
    return out


_MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Base sklearn families.
    "linear": {
        "task": "regression",
        "display_name": "Linear Regression",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "ridge": {
        "task": "regression",
        "display_name": "Ridge Regression",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "elastic_net": {
        "task": "regression",
        "display_name": "Elastic Net Regression",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "rf": {
        "task": "regression",
        "display_name": "Random Forest Regressor",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "gbr": {
        "task": "regression",
        "display_name": "Gradient Boosting Regressor",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "hgbt": {
        "task": "regression",
        "display_name": "HistGradientBoosting Regressor",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "mlp": {
        "task": "regression",
        "display_name": "MLP Regressor",
        "family": "sklearn",
        "supports_native_categorical": False,
        "external_dependency": None,
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    # External gradient boosting families.
    "xgb": {
        "task": "regression",
        "display_name": "XGBoost Regressor",
        "family": "xgboost",
        "supports_native_categorical": False,
        "external_dependency": "xgboost",
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "lgbm": {
        "task": "regression",
        "display_name": "LightGBM Regressor",
        "family": "lightgbm",
        "supports_native_categorical": False,
        "external_dependency": "lightgbm",
        "notes": "Uses shared sklearn preprocessing (median impute + one-hot for categoricals).",
    },
    "catboost": {
        "task": "regression",
        "display_name": "CatBoost Regressor",
        "family": "catboost",
        # CatBoost can support native categoricals, but we currently keep a
        # compatibility-first path (one-hot preprocessing) to avoid destabilizing
        # the architecture.
        "supports_native_categorical": True,
        "external_dependency": "catboost",
        "notes": "Uses shared sklearn preprocessing (one-hot). Internally densifies features for compatibility.",
    },
}


def get_supported_model_specs() -> dict[str, dict[str, Any]]:
    """
    Return a registry of supported model specs.

    The returned dict is primarily for UI / inspection. Builders are provided by
    :func:`build_regression_estimator` and :func:`build_preprocessed_regression_pipeline`.
    """

    # Ensure stable ordering by iterating over `_SUPPORTED_MODEL_NAMES`.
    return {name: dict(_MODEL_SPECS[name]) for name in _SUPPORTED_MODEL_NAMES if name in _MODEL_SPECS}


def _model_hyperparams(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    ms = config.get("model_settings", {}) or {}
    params = ms.get(model_name, {}) or {}
    if not isinstance(params, dict):
        return {}
    return params


def build_regression_estimator(model_name: str, *, random_state: int, config: dict[str, Any]) -> Any:
    """
    Build an *unfitted* sklearn regression estimator by registry name.

    Hyperparameters are read from ``config["model_settings"][model_name]``.
    """

    mn = (model_name or "").strip().lower()
    ms = config.get("model_settings", {}) or {}
    params = _model_hyperparams(config, mn)

    if mn == "linear":
        return LinearRegression()

    if mn == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)))

    if mn == "elastic_net":
        return ElasticNet(
            alpha=float(params.get("alpha", 1.0)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            random_state=random_state,
            max_iter=int(params.get("max_iter", 1000)),
            tol=float(params.get("tol", 1e-4)),
        )

    if mn == "rf":
        n_estimators = int(params.get("n_estimators", ms.get("rf_n_estimators", 200)))
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
        )

    if mn == "gbr":
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )

    if mn == "hgbt":
        # HistGradientBoostingRegressor uses max_iter instead of n_estimators.
        return HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", params.get("n_estimators", 200))),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=params.get("max_depth", None),
            random_state=random_state,
        )

    if mn == "mlp":
        hidden_layer_sizes = params.get("hidden_layer_sizes", (64, 32))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=int(params.get("max_iter", 500)),
            alpha=float(params.get("alpha", 1e-4)),
            learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
            random_state=random_state,
        )

    if mn == "xgb":
        if _XGBRegressor is None:
            raise ExternalModelDependencyMissingError(mn, _EXTERNAL_MODEL_DEPENDENCIES["xgb"])
        # Conservative defaults for tabular regression.
        return _XGBRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            objective=params.get("objective", "reg:squarederror"),
            random_state=random_state,
            verbosity=int(params.get("verbosity", 0)),
        )

    if mn == "lgbm":
        if _LGBMRegressor is None:
            raise ExternalModelDependencyMissingError(mn, _EXTERNAL_MODEL_DEPENDENCIES["lgbm"])
        return _LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=params.get("max_depth", 5),
            num_leaves=int(params.get("num_leaves", 31)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            random_state=random_state,
            n_jobs=params.get("n_jobs", -1),
        )

    if mn == "catboost":
        if _CatBoostRegressor is None:
            raise ExternalModelDependencyMissingError(mn, _EXTERNAL_MODEL_DEPENDENCIES["catboost"])
        return _CatBoostRegressor(
            iterations=int(params.get("iterations", 100)),
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            loss_function=params.get("loss_function", "RMSE"),
            random_seed=random_state,
            verbose=params.get("verbose", False),
            allow_writing_files=params.get("allow_writing_files", False),
        )

    raise ValueError(
        f"Unknown regression model name: {model_name!r}. Supported: {', '.join(_SUPPORTED_MODEL_NAMES)}"
    )


def build_preprocessed_regression_pipeline(
    model_name: str,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
    random_state: int = 0,
) -> Pipeline:
    """
    Build a shared preprocess + model pipeline for regression.

    Preprocessing is consistent across all supported models:
    - numeric: median imputation
    - categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore")
    """

    mn = (model_name or "").strip().lower()
    feature_numeric = [str(c) for c in numeric_features]
    feature_categorical = [str(c) for c in categorical_features]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_numeric),
            ("cat", categorical_transformer, feature_categorical),
        ],
        remainder="drop",
    )

    estimator = build_regression_estimator(model_name, random_state=random_state, config=config)

    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]

    def _make_external_matrix_normalizer() -> FunctionTransformer:
        # Unify the transformed container passed into external estimators.
        #
        # Some sklearn wrappers (notably LightGBM) perform strict feature-name
        # validation and can warn when fit/predict receive different
        # "feature-name aware" containers. To keep things stable, we convert
        # to a dense matrix and return a DataFrame with deterministic column
        # names.
        def _normalize(X: Any) -> Any:
            if hasattr(X, "toarray"):
                dense = np.asarray(X.toarray(), dtype=float)
            else:
                dense = np.asarray(X, dtype=float)

            # Stable "Column_0..N" names so predict() has valid names too.
            cols = [f"Column_{i}" for i in range(dense.shape[1])]
            return pd.DataFrame(dense, columns=cols)

        return FunctionTransformer(_normalize, validate=False)

    if mn in {"xgb", "lgbm", "catboost"}:
        steps.append(("external_matrix_normalizer", _make_external_matrix_normalizer()))

    steps.append(("model", estimator))
    pipe = Pipeline(steps=steps)

    # Some environments configure sklearn to emit pandas outputs from transformers.
    # For external boosted models we want a consistent ndarray container so
    # feature-name checks don't warn on predict().
    try:  # pragma: no cover - depends on sklearn version/config.
        pipe.set_output(transform="default")
    except Exception:
        pass

    return pipe

