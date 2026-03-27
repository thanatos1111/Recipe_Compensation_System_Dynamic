Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a report-generation feature that explains the core model, scoring method, and recommendation generation logic inside the app.

Goal:
Users should be able to understand how recommendations are produced from history data and trained models.

Implement the following:
1. Add a report generator module that produces a structured report for the currently selected material and active target instance
2. The report should explain:
   - input data used
   - target instances included
   - feature columns used
   - model types used
   - training data size and coverage
   - validation method
   - model metrics
   - optimizer scoring formula
   - constraint handling
   - safe-band method
   - recommendation generation pipeline
3. Add both:
   - an in-app readable report view
   - export to markdown or HTML if practical
4. Use real live config values and artifacts rather than static text
5. Include a “current limitations” section:
   - sparse data
   - extrapolation risk
   - machine-state drift not fully modeled if applicable
   - coupled-parameter assumptions

Implementation guidance:
- Keep report-building logic reusable in `core/`
- Keep exported text readable for engineering review
- Do not hardcode outdated design assumptions; use the current app state

Output:
- Add a report feature
- Make model and scoring internals inspectable from the app