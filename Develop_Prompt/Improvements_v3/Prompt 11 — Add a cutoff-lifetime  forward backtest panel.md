Implement a dedicated backtest panel for future-like validation using cutoff lifetime or target selection. Do not change unrelated behavior.

Goal:
Let the user simulate real usage:
- use earlier data as history
- use later data as future
- compare predictions against actual future results

Requirements:
1. Add a new panel/sub-tab named something like Backtest or Forward Validation.
2. Inputs:
   - target variable
   - cutoff lifetime
   - optional target selection / target holdout
   - optional filters such as material/recipe family if already available in app
3. Behavior:
   - train on data before cutoff
   - evaluate on data after cutoff
   - optionally support rolling/expanding window mode later; basic single cutoff first
4. Outputs:
   - metrics summary
   - sample counts train/test
   - actual vs predicted plots
   - residual plots
5. Keep the result object reusable for later trend/recommendation views.

Acceptance criteria:
- I can choose a cutoff lifetime and see how the model would have performed on later data.
- Metrics and plots reflect future-style validation, not random split only.