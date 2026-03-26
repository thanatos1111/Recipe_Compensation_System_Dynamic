Implement Milestone 4 for the `recipe_comp_system` project: material-specific model training for RS, thickness, and RSU, plus current target-instance correction.

Scope:
- Models must be trained independently per selected material.
- Do not pool data across different materials.
- Use all rows from the selected material, including out-of-spec rows.
- Preserve target-instance identity.
- Provide metrics and confidence notes.

Implement the following:

1. In `core/feature_engineering.py`:
   - build feature matrix from canonical material dataframe
   - include lifetime, incident_angle, linear_offset, rotations, o2_flow or o2_ratio, total_flow if needed
   - support target-instance-aware features or grouped handling
   - ensure consistent feature schema storage
2. In `core/response_models.py`:
   - `train_rs_model(...)`
   - `train_thickness_model(...)`
   - `train_rsu_model(...)`
   - optional `train_spec_classifier(...)`
   - `predict_outputs(model_bundle, X)`
3. In `core/instance_correction.py`:
   - fit a light correction for the active target instance based on residuals from the material-level model
   - provide a function to apply that correction to predictions
4. Use practical default models:
   - start with Gradient Boosting Regressor or Random Forest Regressor
   - add fallback to simpler models if row count is too low
5. Validation:
   - use time-aware or grouped validation where practical
   - avoid only random split metrics
6. Store results in artifacts:
   - trained material models
   - feature schema
   - training metrics
   - row count summary
   - confidence summary
   - active-instance correction summary
7. In `ui/model_panel.py`:
   - add controls to train models for selected material
   - display key metrics for RS, thickness, and RSU
   - show active target-instance correction summary
   - show warnings for low-data conditions
8. Add tests for:
   - per-material training isolation
   - target-instance-aware feature handling
   - model artifact creation
   - basic prediction call
   - instance correction output

Important design rules:
- Never train on multiple materials together.
- Out-of-spec rows are allowed and useful for response modeling.
- Keep model training logic in `core/`, not UI.
- Expose low confidence clearly when data is sparse or lifetime coverage is poor.

Output:
- Working per-material model training pipeline.
- Active-instance correction workflow.
- Model metrics panel in UI.
- Summary of model choices, fallback logic, and assumptions.