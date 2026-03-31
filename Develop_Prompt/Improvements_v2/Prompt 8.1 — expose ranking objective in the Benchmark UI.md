Add ranking-objective controls to the Benchmark UI in `Recipe_Compensation_System_Dynamic`.

Goal:
Let the user choose how benchmark winners are ranked, instead of always using spec-pass then RS MAE.

Implement the following:

A. Update:
- `ui/model_panel.py`

B. In the Benchmark setup area, add:
1. a ranking objective dropdown
   - options:
     - spec_pass_first
     - rs_first
     - thickness_first
     - rsu_first
     - weighted_combined
2. a weighted-score config area that is only visible when `weighted_combined` is selected

C. Add weight controls for weighted_combined:
- spec_pass_accuracy weight
- rs_mae weight
- thickness_mae weight
- rsu_mae weight

Use sensible defaults and make them editable with spin boxes.

D. Wire benchmark ranking to the selected objective:
1. when benchmark completes, use the chosen objective for ranking
2. update:
   - winner selection
   - best explanation panel
   - adopt winner behavior
3. show the active ranking objective clearly in the best explanation panel

E. Requirements:
1. Keep existing benchmark execution unchanged.
2. Do not add uncertainty-aware ranking yet in this prompt.
3. Preserve current default behavior by making `spec_pass_first` the default objective.
4. If weighted_combined is selected but all weights are zero, show a readable warning and do not crash.

F. Add tests where practical:
- helper/state tests only if easy
- otherwise ensure backend ranking tests cover most logic

Output:
- summarize where the ranking selector appears
- explain how the chosen objective affects winner selection