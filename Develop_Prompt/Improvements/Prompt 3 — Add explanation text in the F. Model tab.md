Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Improve the F. Model tab by adding explanation panels that describe what the displayed data, metrics, and plots mean.

Requirements:
1. Add a structured explanation area in the model tab
2. For each major item shown in the tab, add plain-language descriptions:
   - training row count
   - in-spec / out-of-spec counts
   - feature list
   - model type
   - validation method
   - MAE / RMSE / other metrics
   - feature importance
   - residual plots
   - confidence summary
3. Add tooltip or expandable help text for each metric
4. Add a summary paragraph such as:
   - what the model is learning
   - what data it used
   - what the confidence means
   - what limitations exist if data is sparse
5. Make the explanations dynamic:
   - if data is sparse, say so
   - if extrapolating beyond the lifetime range, warn clearly
   - if model fallback is being used, explain that too

Implementation guidance:
- Keep explanation generation logic separated from UI rendering
- Add helper functions that turn model artifacts into human-readable descriptions
- Avoid generic placeholder text; use real model state and metrics

Output:
- Improve the model tab with meaningful explanations
- Show exactly how each displayed metric should be interpreted