Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a “manual parameter override / what-if tuning” workflow.

Goal:
Allow the user to manually force one or more parameters to chosen values, then let the system find how the remaining parameters should be tuned to bring predicted results back into spec.

Implement the following:
1. In the recommendation panel, add a manual override mode
2. Let the user:
   - choose one or more parameters
   - lock them to specific values
   - optionally lock multiple parameters at once
3. Then run the optimizer only over the remaining free parameters
4. Respect all existing constraints:
   - discrete step size
   - min/max
   - categorical values
   - coupled parameter rules such as Ar + O2 total flow
5. For coupled parameters:
   - if the user changes one coupled variable, automatically update the coupled dependent variable if appropriate
   - clearly show the coupling logic in the UI
6. Show:
   - locked parameters
   - free parameters
   - recommended adjusted recipe
   - predicted RS / Thickness / RSU
   - whether the locked choice makes in-spec recovery impossible
7. If no feasible solution exists, show that clearly and provide nearest-best alternatives

Implementation guidance:
- Reuse existing candidate generation and scoring logic
- Add a constrained search mode for partially locked recipes
- Add tests for locked-variable optimization and coupled-variable handling

Output:
- Add manual override optimization
- Let user force some values and solve the rest