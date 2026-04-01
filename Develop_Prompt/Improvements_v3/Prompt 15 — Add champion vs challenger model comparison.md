Implement a champion-vs-challenger workflow for safe model updates. Do not change unrelated behavior.

Goal:
Do not replace the production recommendation model blindly whenever new data arrive.

Requirements:
1. Define:
   - champion model = current active model for recommendations
   - challenger model = newly trained candidate
2. Add a comparison view using the same recent validation window for both models.
3. Compare at least:
   - MAE / RMSE
   - in-spec hit rate if available
   - recommendation stability
   - feasibility compliance
4. Add a controlled promote action:
   - keep champion if challenger is not clearly better
   - promote challenger when chosen by user
5. Show the current active model clearly in UI.
6. Persist model metadata cleanly.

Acceptance criteria:
- I can compare old vs new model before switching.
- The app supports safe model evolution instead of blind overwrite.