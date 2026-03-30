Add recommendation backtesting so the app can evaluate whether a model’s recommendations would likely improve real process outcomes, not just prediction error.

Goals:
- Replay historical rows
- Compare baseline recipe vs recommended recipe under the benchmarked model
- Measure practical usefulness of the recommendation engine

Implement the following:

A. Create:
- `core/recommendation_backtest.py`

Add:
1. `run_recommendation_backtest(df, *, config, model_bundle_names, split_mode) -> RecommendationBacktestResult`
2. `backtest_single_fold(train_df, test_df, *, config, model_bundle_name) -> dict`
3. `score_recommendation_outcome(actual_row, recommended_candidate, predicted_outputs, spec_config) -> dict`

B. Backtest logic:
For each fold:
1. train model only on fold train rows
2. for each eligible test row:
   - treat observed recipe as baseline
   - generate recommendation using only train-fitted model
   - compare:
     - baseline predicted score
     - recommended predicted score
     - actual test row spec result
3. aggregate:
- rows evaluated
- recommendation improvement rate
- predicted spec-pass improvement
- move size statistics
- fraction of conservative/no-change recommendations

C. Add UI support in Model tab benchmark area:
- Recommendation Backtest sub-section
- result table + simple plots

D. Add tests:
- `tests/test_recommendation_backtest.py`

Design rules:
- Never use future rows to fit recommendation models
- Keep replay strictly fold-based
- Make outputs usable for comparing future model methods fairly

Output:
- Add recommendation backtesting module
- Wire it to benchmark flow
- Add tests