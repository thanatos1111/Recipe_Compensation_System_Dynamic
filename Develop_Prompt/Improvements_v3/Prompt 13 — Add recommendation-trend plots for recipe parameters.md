Implement recommendation trend plots that compare actual successful recipe parameters against model-recommended parameters over lifetime. Do not change unrelated behavior.

Goal:
For a selected process parameter, show how the recommended setting evolves versus the actual historical in-spec settings.

Requirements:
1. Add a new view in the backtest/recommendation analysis area.
2. Inputs:
   - parameter to plot on y-axis
   - x-axis = lifetime or target/run order
   - cutoff/backtest selection
3. Plot:
   - actual historical in-spec parameter points
   - predicted/recommended parameter points
   - optional connecting lines / smooth trend line
4. Use feasibility-constrained recommendations only.
5. Make it clear which points are:
   - observed actual successful data
   - model recommendation
6. Add an option to show only in-spec actual points, since that is the key comparison.

Acceptance criteria:
- I can choose a parameter like power, pressure, linear_offset, or incident_angle and see actual vs recommended trends over lifetime.
- The plot helps judge whether recommendations are stable and realistic.