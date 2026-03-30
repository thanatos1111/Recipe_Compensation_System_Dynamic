Add a model registry system so the app can benchmark multiple modeling methods through one common interface.

Goals:
- Centralize model creation
- Allow per-response model selection for RS, Thickness, and RSU
- Make future additions easy (CatBoost, XGBoost, LightGBM, MLP, uncertainty variants)

Implement the following:

A. Create:
- `core/model_registry.py`

Add:
1. `get_supported_model_specs() -> dict`
2. `build_regression_estimator(model_name: str, *, random_state: int, config: dict) -> Any`
3. `build_preprocessed_regression_pipeline(model_name: str, *, numeric_features: list[str], categorical_features: list[str], config: dict) -> Pipeline`

Support these model names first:
- `linear`
- `ridge`
- `elastic_net`
- `rf`
- `gbr`
- `hgbt`
- `mlp`

Use sklearn only for this milestone.

B. Update `core/response_models.py`
1. Replace hardcoded model selection logic with the new registry
2. Allow per-target config keys:
- `rs_model`
- `thickness_model`
- `rsu_model`
3. Store chosen model names in training artifacts and benchmark summaries

C. Update config:
- extend `config/default_config.json`
Add reasonable defaults and hyperparameter blocks for:
- ridge
- elastic_net
- random forest
- gradient boosting
- hist gradient boosting
- mlp

D. Add tests:
- `tests/test_model_registry.py`
Test:
1. all declared model names build successfully
2. pipelines can fit a small dummy dataframe
3. unknown model name raises a clear error

Design rules:
- Do not add external packages yet
- Keep feature preprocessing shared and consistent across models
- Preserve target_id categorical handling
- Keep this registry compatible with the leakage-free benchmark evaluator from the previous milestone

Output:
- Add model registry
- Wire current training code to use it
- Update config
- Add tests