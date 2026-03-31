Implement a focused polish/fix pass for the auto-decision engine in `Recipe_Compensation_System_Dynamic`.

Goals:
1. Make the `recommendation-first` auto-decision path actually rank by recommendation backtest usefulness when backtest is feasible.
2. Show skipped bundle reasons in the auto-decision report, in addition to skipped split reasons.
3. Keep manual benchmark unchanged.
4. Keep auto decision as a separate workflow.

Implement the following:

A. Fix recommendation-first logic in backend
Files:
- `core/auto_decision.py`
- optionally `core/auto_decision_goals.py` if naming/description should be clarified

Problem:
- Current auto-decision flow still uses benchmark ranking as the primary order.
- Backtest is only acting as a tie-break in `score_auto_decision_candidates(...)`.
- That means the `recommendation_first` preset is not truly recommendation-first.

Requirements:
1. Detect when the active auto-decision goal is recommendation-oriented.
2. When recommendation backtest is available and feasible, rank candidates primarily by backtest usefulness for recommendation-first mode.
3. Keep a deterministic fallback if backtest is missing, skipped, aborted, or has too few usable rows.

Recommended ranking behavior for recommendation-first:
Primary:
- higher `backtest_predicted_spec_pass_improvement_rate`
Secondary:
- higher `backtest_recommendation_improvement_rate`
Tertiary:
- lower `backtest_no_change_fraction`
Quaternary fallback:
- existing benchmark ranking order

Alternative acceptable implementation:
- add a dedicated `decision_mode` / `goal_mode` argument to `score_auto_decision_candidates(...)`
- use it to switch between:
  - benchmark-first
  - recommendation-first

Do not invent a huge new scoring system in this prompt.
Keep it simple, explicit, and explainable.

B. Pass goal preset / decision mode cleanly into auto-decision scoring
Files:
- `core/auto_decision.py`
- `ui/model_panel.py`
- optionally `core/auto_decision_goals.py`

Requirements:
1. Ensure the selected auto goal preset is available to backend scoring.
2. The backend should know whether the run is:
   - balanced
   - prediction-first
   - recommendation-first
   - conservative-safe
   - weighted_balanced
3. Do not infer recommendation-first only from ranking objective, because currently that preset may still use `spec_pass_first`.
4. A simple config key like:
   - `auto_decision.goal_preset`
   is acceptable.

C. Improve auto-decision report: show skipped bundle reasons too
Files:
- `ui/model_panel.py`

Requirements:
1. In the auto-decision result report, keep the existing skipped split modes section.
2. Add a new section:
   - `Skipped bundles:`
3. Show excluded bundle reasons from `out.excluded_reasons` for keys starting with:
   - `bundle:`
   - and any bundle-filter fallback keys such as `bundles:...`
4. Keep formatting readable and compact.
5. Do not remove the split section.

Suggested rendering:
- `Skipped split modes:`
  - `leave_one_target_out: not_feasible_in_regime`
- `Skipped bundles:`
  - `xgb_default: not_in_allowlist`
  - `catboost_default: denylisted`
  - `bundles:empty_after_filters -> fell_back_to_default_subset`

D. Improve explanation text for recommendation-first runs
Files:
- `core/auto_decision.py`

Requirements:
1. If recommendation-first ranking used backtest as the primary driver, make the explanation explicitly say so.
2. Include the winner’s relevant backtest metrics when available:
   - predicted spec-pass improvement rate
   - recommendation improvement rate
   - rows evaluated
3. If backtest was not feasible and the system fell back to benchmark-first behavior, say that clearly in the explanation.

Example:
- `recommendation-first goal requested; backtest was feasible, so winner ranking prioritized predicted spec-pass improvement rate`
or
- `recommendation-first goal requested, but backtest was unavailable, so ranking fell back to benchmark results`

E. Keep confidence logic intact unless needed for clarity
Files:
- `core/auto_decision.py`

Requirements:
1. Do not redesign confidence estimation in this prompt.
2. But if recommendation-first ranking is used, allow the explanation to mention whether confidence is limited by low backtest rows.

F. Tests
Files:
- `tests/test_auto_decision.py`
- add a new small test file if needed

Add/update tests for:
1. recommendation-first uses backtest metrics as primary ranking when backtest summaries are present
2. recommendation-first falls back safely when backtest is missing
3. skipped bundle reasons appear in rendered auto report helper output, or in helper formatting logic if you extract a formatter
4. explanation text states whether recommendation-first used backtest or fallback behavior

Implementation note:
If UI text rendering is too large to test directly, extract a small pure helper function for report formatting and test that.

Constraints:
- Do not change manual benchmark ranking
- Do not redesign the full auto-decision engine
- Do not auto-train
- Do not auto-adopt without button click
- Keep this as a targeted fix/polish pass

Output:
- summarize changed files
- explain how recommendation-first now differs from benchmark-first
- explain where skipped bundle reasons are shown in the auto report