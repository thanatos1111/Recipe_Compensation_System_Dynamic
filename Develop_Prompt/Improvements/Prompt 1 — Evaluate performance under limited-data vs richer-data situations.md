Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

I want to add a formal evaluation workflow to the app to answer this question:

How do prediction and recommendation performance behave in different data sufficiency situations?

Required scenarios:
1. Case 1: only the first target instance of a material is available, and data is limited.
2. Case 2: historical data from the first target instance is available, plus partial data from the second target instance from lifetime zero to its current lifetime.

Implement a new evaluation module and UI section that compares model and recommendation performance across these scenarios.

Requirements:
1. Add an evaluation mode that can simulate staged data availability for one selected material:
   - Scenario A: train using only target instance 1 with limited rows
   - Scenario B: train using full target instance 1 + partial target instance 2 up to a selected lifetime cutoff
2. For each scenario, compute:
   - prediction error for RS, Thickness, RSU
   - in-spec classification quality if available
   - recommendation success proxy, such as whether recommended recipe candidates are closer to actual in-spec outcomes than baseline/current recipe
3. Add time-aware evaluation:
   - train only on historical rows available up to a cutoff
   - test on later rows not included in training
4. Expose metrics such as:
   - MAE / RMSE for RS
   - MAE / RMSE for Thickness
   - MAE / RMSE for RSU
   - spec pass-rate prediction quality
   - confidence level summary
5. Add visual comparison between scenario A and scenario B
6. In the UI, add a new tab or subsection in the model/evaluation area:
   - choose material
   - choose active target instance
   - choose cutoff lifetime for partial-data simulation
   - run evaluation
   - show metric tables and plots
7. Add a short plain-language interpretation section:
   - whether the model is data-limited
   - whether adding partial current-target data improves performance
   - whether recommendation confidence should be reduced

Implementation guidance:
- Keep different materials isolated.
- Same-material target instances may be used together.
- Use target-instance boundaries explicitly.
- Prefer reusable evaluation functions in `core/`.
- Add tests for both evaluation scenarios.

Output:
- Implement the evaluation workflow
- Add UI controls and plots
- Summarize the metrics and assumptions