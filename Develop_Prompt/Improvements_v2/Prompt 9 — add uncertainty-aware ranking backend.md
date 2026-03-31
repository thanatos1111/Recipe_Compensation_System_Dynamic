Add uncertainty-aware benchmark ranking in `Recipe_Compensation_System_Dynamic`.

Goal:
Allow ranking to optionally include interval-quality information, so overconfident models are not treated the same as well-calibrated models.

Implement the following:

A. Update:
- `core/ranking.py`

B. Add uncertainty ranking modes:
1. `ignore`
2. `warn_only`
3. `include_in_score`

C. For uncertainty-aware scoring, use available benchmark uncertainty metrics:
- interval coverage
- interval mean width
- optionally median width

D. Add logic:
1. derive target coverage from conformal alpha when available
2. compute uncertainty quality per response using:
   - coverage closeness to target
   - width penalty
3. aggregate uncertainty quality into bundle-level ranking when `include_in_score` is selected

E. Requirements:
1. work even when uncertainty metrics are missing
2. do not crash if uncertainty was disabled during benchmark
3. in `warn_only` mode:
   - do not change ranking score
   - but return warning/explanation text if coverage looks poor
4. in `include_in_score` mode:
   - uncertainty should influence weighted_combined and optionally other objectives as a secondary adjustment

F. Add/update APIs:
- extend ranking functions to accept:
  - `uncertainty_mode`
  - `uncertainty_weights` or equivalent settings if needed

G. Add tests:
- `tests/test_ranking_uncertainty.py`

Test:
1. missing uncertainty metrics do not crash ranking
2. better calibrated bundles score better when uncertainty is included
3. warn_only produces readable warning metadata without changing rank order

Output:
- summarize uncertainty ranking modes
- explain how uncertainty now affects scoring