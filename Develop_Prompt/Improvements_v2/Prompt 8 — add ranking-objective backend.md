Implement a ranking-objective system for benchmark results in `Recipe_Compensation_System_Dynamic`.

Goal:
Replace the current fixed benchmark ranking logic with a reusable objective-driven ranking backend.

Context:
- The current benchmark runner can already evaluate multiple bundles and rank them.
- Right now ranking is effectively hard-coded around:
  - primary = spec_pass_accuracy
  - secondary = rs_mae
- We need to support user-selectable ranking objectives before building auto model selection.

Implement the following:

A. Create a new module:
- `core/ranking.py`

B. Add supported ranking objectives:
1. `spec_pass_first`
2. `rs_first`
3. `thickness_first`
4. `rsu_first`
5. `weighted_combined`

C. Add APIs:
1. `get_supported_ranking_objectives() -> dict`
   - return UI-friendly metadata:
     - display_name
     - description
     - whether custom weights are used
2. `rank_benchmark_suite(results, *, objective: str, weights: dict | None = None, uncertainty_mode: str = "ignore") -> list[dict]`
3. `score_benchmark_bundle(run_summaries, *, objective: str, weights: dict | None = None, uncertainty_mode: str = "ignore") -> dict`

D. Ranking behavior:
1. `spec_pass_first`
   - prioritize higher spec-pass accuracy
   - tie-break with lower RS MAE
2. `rs_first`
   - prioritize lower RS MAE
   - tie-break with higher spec-pass accuracy
3. `thickness_first`
   - prioritize lower Thickness MAE
   - tie-break with higher spec-pass accuracy
4. `rsu_first`
   - prioritize lower RSU MAE
   - tie-break with higher spec-pass accuracy
5. `weighted_combined`
   - compute a normalized weighted score from summary metrics
   - support weights for:
     - spec_pass_accuracy
     - rs_mae
     - thickness_mae
     - rsu_mae
   - lower error metrics should improve score after proper direction handling

E. Requirements:
1. Keep current benchmark runner unchanged as much as possible.
2. Do not add UI in this prompt.
3. Preserve deterministic ranking.
4. Make the new ranking layer reusable by manual benchmark and future auto-decision engine.
5. Keep bundle aggregation across split modes supported.

F. Update existing code:
- `core/benchmarking.py`
Use the new ranking module instead of hard-coding ranking logic inside `rank_benchmark_results(...)`.
You may keep `rank_benchmark_results(...)` as a compatibility wrapper if useful, but route it through `core/ranking.py`.

G. Add tests:
- `tests/test_ranking.py`

Test:
1. each supported objective returns a deterministic sorted result
2. weighted_combined works with provided weights
3. lower errors improve the correct objectives
4. higher spec-pass improves the correct objectives
5. missing metrics are handled gracefully without crashing

Output:
- summarize new files and changed files
- explain how ranking objectives are now separated from benchmark execution