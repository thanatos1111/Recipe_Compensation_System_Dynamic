Add an Auto Decision section to the Benchmark UI in `Recipe_Compensation_System_Dynamic`.

Important:
This must be separate from the current manual benchmark workflow.
Users must still be able to run benchmark manually as before.

Goal:
Let users run an automatic regime-aware model-selection workflow and configure how the app decides the winner.

Implement the following:

A. Update:
- `ui/model_panel.py`

B. Add a new subsection in the Benchmark tab:
Title:
- `Auto decision engine`

C. Add controls for user goals:
1. ranking objective dropdown
   - same objectives as manual benchmark
2. uncertainty mode selector
   - ignore
   - warn_only
   - include_in_score
3. auto-decision goal presets, such as:
   - prediction-first
   - recommendation-first
   - conservative-safe
   - balanced
4. if weighted_combined is selected:
   - reuse or mirror weighted score controls
5. allow bundle scope selection:
   - all available bundles
   - selected bundles only

D. Add run button:
- `Run auto model selection`

E. Add result display:
Show:
- detected data regime
- split modes used
- skipped split modes and why
- bundles evaluated
- winner
- runner-up
- confidence level
- explanation
- suggested next action:
  - adopt winner
  - train winner now (button can be added later if needed)
  - keep manual review

F. Add buttons:
1. `Adopt auto-selected winner`
2. optional `Copy auto decision report`

G. Requirements:
1. Keep manual benchmark controls intact.
2. Auto decision should not overwrite manual benchmark results unless explicitly intended.
3. Auto decision should not retrain automatically in this prompt.
4. Make it obvious that this is a separate automated assistant workflow.

H. If helpful, reuse existing best-explanation and warning-summary patterns.

I. Add tests only for helper/state logic if practical.

Output:
- summarize where the auto-decision UI appears
- explain how users configure goals for automatic model selection