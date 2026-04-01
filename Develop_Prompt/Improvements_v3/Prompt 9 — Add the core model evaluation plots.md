Implement the first set of model evaluation plots in the training/evaluation area. Do not change unrelated behavior.

Goal:
Add standard plots that make model quality visible.

Required plots:
1. Predicted vs Actual scatter
2. Residual vs Predicted
3. Residual vs Lifetime
4. Actual vs Predicted over Lifetime

Requirements:
1. Add a new evaluation/visualization section or sub-tab in the training area.
2. Let the user choose:
   - target variable
   - evaluation split result
   - train/validation/future subset where applicable
3. Plot requirements:
   - clean axes labels
   - legend where needed
   - 45-degree reference line for Predicted vs Actual
   - residual zero line where appropriate
4. Use the app’s plotting stack consistently.
5. Add hover/value readout if the app already has a hover pattern; otherwise keep it simple and stable.
6. Make sure the plots work with lifetime-based evaluation results from Prompt 8.

Acceptance criteria:
- I can visually inspect actual vs predicted quality.
- I can see residual trends over lifetime.
- The plots update when a different target/evaluation result is selected.