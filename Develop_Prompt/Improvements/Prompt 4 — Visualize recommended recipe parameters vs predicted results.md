Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add visualization of the relationship between recommended recipe parameters and predicted results.

Goal:
When the app recommends a recipe, the user should be able to see how recipe values relate to predicted RS, Thickness, and RSU.

Implement the following:
1. In the recommendation panel, add a visualization area that shows:
   - recommended recipe parameter values
   - predicted RS
   - predicted Thickness
   - predicted RSU
   - score breakdown
2. Add at least two plot styles:
   - a recipe summary chart/table for the selected best candidate
   - a candidate comparison plot for top-ranked candidates
3. Show, for top N candidates:
   - each parameter set
   - predicted outputs
   - score
4. Add a plot or interactive view showing parameter changes versus predicted output changes
5. Make it clear which candidate is the chosen recommendation
6. If practical, allow the user to select a different candidate and compare it side-by-side with the top candidate

Implementation guidance:
- Use the already computed candidate table
- Keep plots interpretable and compact
- Do not change the optimizer logic yet unless needed for visualization

Output:
- Add recommendation-result visualization
- Make recipe-to-prediction relationship easy to inspect