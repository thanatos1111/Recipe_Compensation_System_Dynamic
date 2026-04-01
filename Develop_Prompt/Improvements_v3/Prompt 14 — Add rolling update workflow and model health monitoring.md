Implement a real update workflow so the app can be used continuously as new deposition data arrive. Do not change unrelated behavior.

Goal:
Move from one-time training to rolling model maintenance.

Requirements:
1. Extend the update area (or H.Update equivalent) with:
   - newly added record summary
   - retrain trigger action
   - recent validation summary
   - drift / stability indicators
2. Add rolling monitoring metrics:
   - recent MAE / RMSE
   - recent in-spec hit rate if available
   - recommendation drift magnitude versus previous model
3. Add a simple model version history:
   - version ID
   - training date
   - data range used
   - chosen model type
   - key metrics
4. Add a visible status summary:
   - stable
   - warning
   - retrain recommended
5. Keep the UX practical for regular use after each new deposition batch.

Acceptance criteria:
- After adding new data, I can evaluate whether the current model is still healthy.
- I can see when retraining is recommended.
- Model version history is available.