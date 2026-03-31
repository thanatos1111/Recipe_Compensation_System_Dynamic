Add a separate auto-decision engine backend for model selection in `Recipe_Compensation_System_Dynamic`.

Important:
This must be a separate function in addition to the current manual benchmark workflow.
Do not replace the current manual benchmark process.

Goal:
Build a regime-aware backend that can automatically evaluate candidate bundles and recommend a deployment winner based on user-configurable goals.

Implement the following:

A. Create:
- `core/data_regime.py`
- `core/auto_decision.py`

B. In `core/data_regime.py`, detect selected-material data conditions:
Return metadata such as:
- total row count
- target count
- rows per target
- whether leave-one-target-out is feasible
- whether active-target-cutoff is feasible
- whether uncertainty calibration is feasible
- whether recommendation backtest is feasible
- regime label, such as:
  - `rich_multi_target`
  - `moderate_multi_target`
  - `single_target_partial`
  - `very_sparse`

C. In `core/auto_decision.py`, add:
1. `build_auto_evaluation_plan(df, config) -> dict`
2. `run_auto_model_selection(df, config) -> AutoDecisionResult`
3. `score_auto_decision_candidates(...)`

D. Auto-decision flow:
1. inspect data regime
2. choose valid split modes automatically
3. choose eligible bundles automatically
4. run benchmark in stages:
   - prediction screening
   - uncertainty screening if feasible
   - recommendation backtest if feasible
5. rank candidates using selected ranking objective and goals
6. return:
   - detected regime
   - split modes used
   - bundles evaluated
   - shortlisted bundles
   - winner
   - runner-up
   - confidence level
   - explanation text
   - reasons for excluded modes/bundles

E. Requirements:
1. Keep it fully separate from manual benchmark.
2. Do not auto-train in this prompt.
3. Do not auto-adopt without explicit UI action.
4. Reuse existing benchmark, uncertainty, and backtest infrastructure.
5. Work with user-selected ranking objective and custom bundles.

F. Add tests:
- `tests/test_data_regime.py`
- `tests/test_auto_decision.py`

Test:
1. correct regime detection for several synthetic cases
2. valid split-mode selection by regime
3. sparse-data cases avoid invalid evaluation paths
4. auto-decision returns structured result without crashing

Output:
- summarize the auto-decision backend flow
- explain how it remains separate from manual benchmark