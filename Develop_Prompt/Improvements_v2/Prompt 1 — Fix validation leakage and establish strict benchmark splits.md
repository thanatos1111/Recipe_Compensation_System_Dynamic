Fix the validation method in this app so benchmark metrics are leakage-free and can serve as the standard evaluation base for all future ML models.

Current problem to fix:
- The current flow trains models on the full material dataframe first, then computes holdout MAE on the same already-fitted models.
- This causes evaluation leakage because test rows have already influenced training.

Goals:
1. Refactor model training and evaluation so that every benchmark score is computed from models fitted only on the training fold of that split.
2. Keep all evaluation material-specific. Never mix data across materials.
3. Preserve target-instance identity and support time-aware and target-aware evaluation.

Implement the following:

A. Create a new module:
- `core/validation_schemes.py`

Add reusable split generators:
1. `make_forward_chaining_splits(df, *, target_id_col="target_id", lifetime_col="lifetime", min_train_rows=10, min_test_rows=3, n_splits=3)`
   - Sort rows by target_id and lifetime
   - Yield multiple train/test index splits where train rows are earlier and test rows are later
   - Do not let test rows overlap with train rows
2. `make_leave_one_target_out_splits(df, *, target_id_col="target_id", min_train_rows=10, min_test_rows=3)`
   - Hold out one target instance at a time as test
   - Use all other same-material target instances as train
3. `make_active_target_cutoff_split(df, *, active_target_id, cutoff_lifetime, target_id_col="target_id", lifetime_col="lifetime")`
   - Train on selected history targets + active target rows up to cutoff
   - Test on active target rows after cutoff

B. Refactor `core/response_models.py`
1. Separate:
   - model construction / pipeline creation
   - fold training
   - fold prediction
   - aggregate evaluation
2. Add:
   - `build_model_pipeline(target_col, config)`
   - `fit_model_for_target(df_train, target_col, config)`
   - `predict_target(model, X_test)`
3. Replace the current optimistic `evaluate_models_time_aware(...)` with a new leakage-free evaluator:
   - `evaluate_models_with_splits(df, *, config, split_strategy)`
   - For each split:
     - build feature matrix from train/test independently using the same feature schema
     - fit models only on train rows
     - predict only on test rows
     - compute fold metrics
   - aggregate fold metrics into mean/std and per-fold records
4. Update `train_material_models(...)`:
   - train final deployment models on all available training rows only for deployment use
   - evaluation metrics must come from the new split-based evaluator, not from the final fitted models directly

C. Add a benchmark result structure in `core/schemas.py`
- `BenchmarkFoldResult`
- `BenchmarkSummary`
Include:
- split_name
- split_type
- train_row_count
- test_row_count
- train_target_ids
- test_target_ids
- rs_mae / rs_rmse
- thickness_mae / thickness_rmse
- rsu_mae / rsu_rmse
- spec_pass_accuracy
- warnings

D. Add tests:
- `tests/test_validation_schemes.py`
- `tests/test_leakage_free_evaluation.py`
Test cases:
1. forward split has no overlap between train/test indices
2. leave-one-target-out really isolates the held-out target
3. benchmark evaluator fits only on train rows
4. benchmark summary returns fold-level and aggregate metrics
5. no cross-material mixing

Design rules:
- Keep the existing UI working.
- Do not add new model types yet.
- Use the current sklearn models first.
- Make the new benchmark evaluator reusable by all future models.

Output:
- Implement the new validation split module
- Refactor response model evaluation to be leakage-free
- Keep final deployment training separate from benchmark evaluation
- Summarize all changed files and the new benchmark flow