Implement Prompt 7 for `Recipe_Compensation_System_Dynamic`: add CatBoost, XGBoost, and LightGBM into the model registry and benchmark runner.

Context:
- The app already has a leakage-free benchmark foundation, a model registry, and a benchmark runner from earlier prompts.
- Right now the dependency set is still sklearn-only.
- This prompt extends the app with optional external gradient boosting models that are strong for tabular regression on process data.

Goals:
1. Add CatBoost, XGBoost, and LightGBM as supported regression model families.
2. Integrate them into the existing model registry.
3. Make them benchmarkable through the same leakage-free evaluation pipeline.
4. Keep the app robust when these packages are not installed.
5. Do not change recommendation logic yet beyond allowing these models to supply predictions through the existing interfaces.

Implement the following:

A. Update dependencies
1. Update `requirements.txt` to add:
   - `xgboost`
   - `lightgbm`
   - `catboost`
2. Keep existing dependencies unchanged.

B. Extend the model registry
1. Update `core/model_registry.py`
2. Add support for these new model names:
   - `xgb`
   - `lgbm`
   - `catboost`
3. Extend:
   - `get_supported_model_specs()`
   - `build_regression_estimator(...)`
   - `build_preprocessed_regression_pipeline(...)`
4. Model behavior requirements:
   - `xgb`: use `xgboost.XGBRegressor`
   - `lgbm`: use `lightgbm.LGBMRegressor`
   - `catboost`: use `catboost.CatBoostRegressor`
5. Add clear metadata to the supported-model spec table:
   - display_name
   - family
   - supports_native_categorical
   - external_dependency
   - notes

C. Installation-safe fallback behavior
1. The app must not crash if one or more external packages are missing.
2. Implement safe import guards in `core/model_registry.py`.
3. If a model is requested but its package is unavailable:
   - raise a clear controlled error in backend-only contexts
   - surface a readable warning in benchmark summaries and UI
4. Add helper(s):
   - `is_model_available(model_name: str) -> bool`
   - `get_model_availability_summary() -> dict`

D. Pipeline design rules for each external model
1. For `xgb` and `lgbm`:
   - keep the existing shared preprocessing pipeline pattern
   - use numeric imputation and one-hot encoding for categorical columns
   - preserve compatibility with the current feature engineering output
2. For `catboost`:
   - support two modes:
     - default mode: use the same preprocessed matrix path as other models for compatibility
     - optional native-categorical mode if feasible without destabilizing the current architecture
3. Do not refactor unrelated modules unless necessary.
4. Keep the training interface identical from the perspective of `core/response_models.py` and `core/benchmarking.py`.

E. Update config
1. Extend `config/default_config.json`
2. Add external model support to model settings:
   - allow `rs_model`, `thickness_model`, `rsu_model` to take values:
     - `xgb`
     - `lgbm`
     - `catboost`
3. Add hyperparameter blocks:
   - `xgb_params`
   - `lgbm_params`
   - `catboost_params`
4. Include reasonable, conservative defaults for tabular regression:
   - moderate tree depth
   - moderate estimator count
   - learning rate defaults
   - random_state where supported
5. Keep the existing sklearn defaults intact.

F. Wire into response model training
1. Update `core/response_models.py`
2. Ensure the new model names work through the existing registry path.
3. Store the selected model names in the returned training artifacts.
4. Preserve the existing per-response configuration pattern:
   - `rs_model`
   - `thickness_model`
   - `rsu_model`

G. Wire into benchmarking
1. Update `core/benchmarking.py`
2. Ensure benchmark suites can include model bundles using:
   - `xgb`
   - `lgbm`
   - `catboost`
3. Add new preset model bundles:
   - `xgb_default`
   - `lgbm_default`
   - `catboost_default`
   - `boosting_compare`
4. Suggested preset definitions:
   - `xgb_default = xgb / xgb / xgb`
   - `lgbm_default = lgbm / lgbm / lgbm`
   - `catboost_default = catboost / catboost / catboost`
   - `boosting_compare` should be a benchmark grouping preset or helper collection that makes these easy to compare against `balanced_default`
5. Benchmark results must include model-availability warnings when relevant.

H. Update benchmark UI
1. Update `ui/model_panel.py`
2. In the Benchmark section:
   - add the new model bundles to the selectable list
   - show model availability status
   - show a warning if a selected bundle cannot run because a dependency is missing
3. Add a small readable availability summary area, for example:
   - sklearn models: available
   - xgb: available / missing
   - lgbm: available / missing
   - catboost: available / missing
4. If a benchmark bundle partially fails because of missing package(s), the UI must:
   - keep other benchmark bundles running
   - report the failure cleanly
   - not crash the whole benchmark tab

I. Add tests
Create or update:
- `tests/test_model_registry_external.py`
- `tests/test_benchmarking_external_models.py`

Test cases:
1. model registry reports availability correctly
2. requesting an unavailable model returns a controlled error
3. external model names are accepted by config parsing
4. benchmark runner skips or flags unavailable models cleanly
5. when packages are installed, each external model can fit a small dummy regression dataset
6. benchmark summaries preserve model names and warnings correctly

J. Design constraints
- Keep all evaluation leakage-free and fold-safe.
- Do not mix materials across training and testing.
- Do not change the recommendation scoring method in this prompt.
- Do not add INN or PyTorch in this prompt.
- Keep the external model integration modular so future uncertainty and backtesting prompts can reuse it directly.
- Preserve backward compatibility with existing sklearn-only workflows.

K. Output requirements
- Update all necessary files
- Summarize:
  - added dependencies
  - new supported model names
  - availability/fallback behavior
  - benchmark presets added
  - any assumptions or limitations