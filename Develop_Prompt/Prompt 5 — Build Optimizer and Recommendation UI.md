Implement Milestone 5 for the `recipe_comp_system` project: discrete constrained optimizer and recommendation UI.

Scope:
- For the currently selected material and active target instance only.
- Use that material’s trained models and the active instance correction.
- Generate valid candidate recipes.
- Predict outputs and rank candidates.
- Display recommended next recipe in the UI.

Implement the following:

1. In `core/constraints.py`:
   - define parameter validity checks
   - enforce min/max bounds
   - enforce discrete step size
   - enforce categorical allowed values
   - enforce coupling rules such as `ar_flow + o2_flow = constant` when configured
2. In `core/candidate_generator.py`:
   - generate local candidate recipes around a reference recipe
   - support using latest actual recipe or baseline compensation recipe as reference
   - keep search bounded and computationally reasonable
3. In `core/optimizer.py`:
   - score candidates using predicted RS, thickness, and RSU
   - include penalties for spec violation and large recipe movement
   - rank candidates and return best option
4. In `core/safe_band.py`:
   - implement a first simple feasible-band estimator around the center recommendation
   - vary one parameter at a time while holding others fixed
   - identify acceptable region where specs are predicted to pass
5. In `ui/recommendation_panel.py`:
   - inputs for current lifetime and current/reference recipe
   - button to generate recommendation for selected material and active target instance
   - show ranked candidate table
   - highlight best candidate
   - show predicted RS / thickness / RSU
   - show change vs reference recipe
   - display confidence or extrapolation warnings
6. Use config-driven optimizer weights from `default_config.json`
7. Add tests for:
   - constraint enforcement
   - gas coupling logic
   - candidate generation validity
   - score ranking behavior
   - recommendation output shape

Important design rules:
- No cross-material mixing.
- Same-material history is allowed through the material model.
- Quantize all numeric recipe outputs before final recommendation.
- Reject invalid candidates before scoring.
- Recommendation must be explainable with score breakdown and predicted outputs.
- Prefer conservative local search over overly complex global optimization in this milestone.

Output:
- Working recommendation workflow for one selected material and active target instance.
- Ranked candidate table and best recipe output in the UI.
- Summary of scoring logic, constraints handled, and remaining gaps.