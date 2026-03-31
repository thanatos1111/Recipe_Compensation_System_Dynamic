Expose uncertainty ranking mode in the Benchmark UI and wire it into winner selection in `Recipe_Compensation_System_Dynamic`.

Goal:
The backend already supports uncertainty-aware ranking modes (`ignore`, `warn_only`, `include_in_score`), but the Benchmark UI does not yet let the user select them. Add a compact UI control and pass the selected mode into benchmark ranking.

Implement the following:

A. Update:
- `ui/model_panel.py`

B. Add Benchmark UI controls
1. In the Benchmark setup area, add a new dropdown:
   - label: `Uncertainty ranking mode`
   - options:
     - `ignore`
     - `warn_only`
     - `include_in_score`
2. Default to:
   - `ignore`
3. Add a small help label or tooltip explaining:
   - `ignore`: rank by normal benchmark metrics only
   - `warn_only`: do not change ranking, but show interval-quality warnings
   - `include_in_score`: include uncertainty quality in ranking

C. Optional uncertainty-weight controls
Only if simple and low-risk:
1. Add one spin box for:
   - `interval_quality` weight
2. Show it only when uncertainty ranking mode is `include_in_score`
3. Default to `1.0`

Do not overbuild this in this prompt.

D. Wire ranking call
In `_benchmark_run_clicked()`:
1. Read the selected uncertainty ranking mode from the new dropdown
2. Pass it into:
   - `rank_benchmark_suite(...)`
3. Also pass:
   - `conformal_alpha` from the existing uncertainty alpha control when uncertainty is enabled
   - `uncertainty_weights` if you added the optional weight control
4. Keep current ranking objective behavior unchanged otherwise

E. Persist UI state in panel memory
Store:
- `_last_benchmark_uncertainty_ranking_mode`
- `_last_benchmark_uncertainty_ranking_weights` if applicable

F. Improve best explanation panel
Update `_render_benchmark_best_explanation(...)` so it shows:
1. selected uncertainty ranking mode
2. if `warn_only`:
   - include any `uncertainty_warnings` from the ranked winner if present
3. if `include_in_score`:
   - show `uncertainty_quality` for winner and runner-up if present
   - show uncertainty weight if applicable

G. Keep behavior safe
1. If uncertainty metrics are unavailable, do not crash
2. If user selects `warn_only` or `include_in_score` but benchmark uncertainty was not enabled, still behave gracefully
3. Keep current benchmark tables unchanged in this prompt

H. Add/update tests
- add a small test file if useful, e.g. `tests/test_ranking_ui_state.py`
- or keep tests focused on helper/state logic if UI tests are too heavy

At minimum test:
1. ranking call accepts selected uncertainty mode
2. `warn_only` does not crash when uncertainty metrics are missing
3. `include_in_score` does not crash when uncertainty is disabled
4. best explanation text includes uncertainty ranking mode

I. Constraints
- Do not redesign the Benchmark layout
- Do not add chart tabs in this prompt
- Do not change benchmark execution logic
- This prompt is only for exposing uncertainty ranking mode and wiring it into winner selection

Output:
- summarize changed files
- explain where the new uncertainty ranking mode control appears
- explain how it now affects winner selection