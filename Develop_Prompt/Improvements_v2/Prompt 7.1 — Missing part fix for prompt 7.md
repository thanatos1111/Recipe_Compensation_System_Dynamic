Implement a cleanup/fix pass for Prompt 7 in `Recipe_Compensation_System_Dynamic`.

Context:
- Prompt 7 was partially realized.
- External models (`xgb`, `lgbm`, `catboost`) are already in the registry and benchmark bundles exist.
- Current issues to fix:
  1. Repeated LightGBM warning during benchmark/backtest:
     `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names`
  2. Missing pieces from the original Prompt 7:
     - no visible model availability summary in the Benchmark UI
     - missing config/default hyperparameter blocks for `xgb`, `lgbm`, `catboost`
  3. Need stronger tests around the warning fix and UI/backend availability behavior.

Goals:
1. Remove the LightGBM feature-name warning by making fit/predict input format consistent.
2. Add the missing external-model availability summary to the Benchmark UI.
3. Extend default config with explicit external-model hyperparameter blocks.
4. Keep all existing benchmark and backtest behavior working.
5. Add tests for the new fixes.

Implement the following:

A. Fix the LightGBM feature-name warning
Files:
- `core/model_registry.py`

Problem:
- LightGBM is currently warning because fit and predict are receiving inconsistent container types / feature-name handling.
- The current pipeline only adds a `to_dense` conversion step for `catboost`.
- Make LightGBM use a stable transformed container too.

Requirements:
1. Add a shared helper transformer for external boosted models that converts transformed output into a consistent dense NumPy array when needed.
2. Apply this normalization step to at least:
   - `lgbm`
   - `catboost`
3. Prefer also applying it to:
   - `xgb`
   for consistency across external models.
4. Keep sklearn-only models unchanged.
5. Do not break sparse-supporting sklearn models unnecessarily.

Implementation guidance:
- Add a helper like:
  - `_make_external_matrix_normalizer(model_name: str) -> Transformer`
- Use `FunctionTransformer(validate=False)` or a small custom transformer class.
- The transformer should:
  - if input has `.toarray()`, convert to dense array
  - otherwise convert to `np.asarray(...)`
- Ensure both `.fit(...)` and `.predict(...)` go through the same normalized path.

Success criteria:
- Benchmark and backtest with `lgbm_default` no longer emit the repeated sklearn/LightGBM feature-name warning.
- External model predictions still work.

B. Add explicit external hyperparameter blocks to default config
Files:
- `config/default_config.json`

Requirements:
1. Add model_settings entries for:
   - `xgb`
   - `lgbm`
   - `catboost`
2. Include conservative defaults, for example:
   - xgb:
     - n_estimators
     - max_depth
     - learning_rate
     - subsample
     - colsample_bytree
     - min_child_weight
     - verbosity
   - lgbm:
     - n_estimators
     - learning_rate
     - max_depth
     - num_leaves
     - subsample
     - colsample_bytree
     - min_child_samples
     - n_jobs
   - catboost:
     - iterations
     - depth
     - learning_rate
     - loss_function
     - verbose
     - allow_writing_files
3. Keep all existing sklearn model settings intact.
4. Do not silently remove old keys.

C. Add external-model availability summary to the Benchmark UI
Files:
- `ui/model_panel.py`

Requirements:
1. In the Benchmark section, add a compact availability summary area for supported model families.
2. Display at least:
   - sklearn models: available
   - xgb: available / missing
   - lgbm: available / missing
   - catboost: available / missing
3. Reuse `get_model_availability_summary()` from `core/model_registry.py`.
4. Show readable fallback/help text for missing packages, e.g.:
   - `missing (install: pip install lightgbm)`
5. Refresh this summary when the Benchmark tab is initialized.
6. If selected bundles include unavailable model families:
   - keep the app running
   - show the warning clearly in the benchmark status / best-explanation / summary warnings
   - allow other bundles to continue

UI guidance:
- A small `QGroupBox("Model availability")` above or near benchmark controls is fine.
- A read-only text block, small table, or form layout is acceptable.
- Keep it compact and readable.

D. Improve benchmark/backtest warning surfacing
Files:
- `ui/model_panel.py`
- maybe supporting helpers if needed

Requirements:
1. When benchmark results contain aggregate warnings for unavailable external models, render them clearly in the UI.
2. If benchmark bundles partially fail, preserve successful bundles.
3. If recommendation backtest is run on unavailable bundles, fail gracefully and report the problem clearly instead of crashing or silently producing confusing empty results.
4. Keep existing progress/abort behavior unchanged.

E. Add tests for the LightGBM warning fix and config/UI integration
Files:
- `tests/test_model_registry_external.py`
- `tests/test_benchmarking_external_models.py`
- add new targeted test file(s) if needed

Tests to add:
1. Pipeline normalization for `lgbm`:
   - build the lgbm pipeline
   - confirm the pipeline contains the external normalization step
2. Pipeline normalization for `xgb` / `catboost` if applied
3. External config keys exist in `default_config.json`
4. Availability summary function returns all external families with expected fields
5. Benchmark still runs when one or more external models are unavailable
6. If LightGBM is installed:
   - fit/predict on a small dummy dataset
   - assert no `X does not have valid feature names` warning is emitted
   Use `warnings.catch_warnings(record=True)` and fail if that specific warning appears.

F. Keep architecture clean
Requirements:
1. Do not redesign the benchmark system.
2. Do not change ranking policy in this prompt.
3. Do not add new model families beyond `xgb`, `lgbm`, `catboost`.
4. Do not remove the current optional-dependency behavior.
5. Keep the solution backward-compatible with sklearn-only use.

G. Output requirements
After implementing, summarize:
- files changed
- how the LightGBM warning was fixed
- where the new availability summary appears in the UI
- what external-model config blocks were added
- any remaining limitations