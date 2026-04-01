Add a training diagnostics backend focused on tabular industrial regression/classification, not deep-learning epoch visuals. Do not change unrelated behavior.

Goal:
Make model training meaningful by adding proper evaluation modes and metrics.

Requirements:
1. Add support for evaluation modes:
   - random split
   - grouped split if target/run grouping exists
   - forward / cutoff-lifetime split
   - target-based holdout if target selection exists
2. Add metrics for regression:
   - MAE
   - RMSE
   - R²
   - optional MAPE only when safe
3. Add metrics for in-spec classification if classification output exists:
   - accuracy
   - precision
   - recall
   - F1
   - confusion matrix counts
4. Store per-run evaluation results in a structured result object.
5. Add support for train vs validation summary reporting.
6. Do not create plots yet unless needed minimally; focus on the backend result structure to feed later visualizations.

Acceptance criteria:
- Training/evaluation can be run in more realistic split modes.
- Metrics are computed and stored cleanly.
- The backend is ready for visual tabs.