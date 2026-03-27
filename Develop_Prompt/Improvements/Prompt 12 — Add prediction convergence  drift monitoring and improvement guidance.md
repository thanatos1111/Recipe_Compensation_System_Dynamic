Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a model monitoring workflow that tells whether the model is predicting well over time, and whether prediction quality is converging or drifting.

Goal:
As new data is added, the app should show whether predictions remain aligned with actual measurements or are getting worse.

Implement the following:
1. Add a prediction-monitoring section
2. For each new appended run, store:
   - predicted RS / Thickness / RSU made before actual result was known
   - actual RS / Thickness / RSU measured later
   - prediction errors
3. Build time-series plots of prediction error over run index / lifetime / date
4. Add convergence and drift plots, such as:
   - rolling MAE
   - rolling bias
   - cumulative error trend
   - residual distribution over time
5. Add rules or flags that detect:
   - improving model fit
   - stable fit
   - degrading fit
   - sudden drift after machine change or target transition
6. Add a guidance section that suggests likely next actions if the model is not performing well, such as:
   - retrain model
   - increase weighting of current target data
   - widen or narrow search space
   - recheck machine event markers
   - inspect feature set
   - review optimizer weights separately from prediction model quality
7. Make it clear that poor prediction quality is not the same thing as poor scoring-term choice:
   - prediction model error and optimizer score design should be shown separately

Implementation guidance:
- Add clear separation between:
   a. prediction model quality
   b. recommendation / scoring behavior
- Use rolling and cumulative diagnostics
- Support segmented diagnostics before/after machine events if those exist

Output:
- Add convergence/drift monitoring
- Show whether the model remains trustworthy as new data arrives
- Provide actionable engineering guidance when it does not