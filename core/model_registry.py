"""
Model registry for recipe parameter regression.

This module centralizes sklearn estimator + preprocessing construction so that
future ML methods can be benchmarked via a consistent interface.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


_SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "linear",
    "ridge",
    "elastic_net",
    "rf",
    "gbr",
    "hgbt",
    "mlp",
)


def get_supported_model_specs() -> dict[str, dict[str, Any]]:
    """
    Return a registry of supported model specs.

    The returned dict is primarily for UI / inspection. Builders are provided by
    :func:`build_regression_estimator` and :func:`build_preprocessed_regression_pipeline`.
    """

    return {name: {"task": "regression"} for name in _SUPPORTED_MODEL_NAMES}


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
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

