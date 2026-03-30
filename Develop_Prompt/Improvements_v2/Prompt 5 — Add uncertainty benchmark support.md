Add uncertainty estimation and uncertainty evaluation so future models are compared not only by point error but also by confidence quality.

Scope for this milestone:
- Start with simple model-agnostic conformal prediction on top of current regression models
- Do not add Bayesian models yet
- Keep this modular for future extension

Implement the following:

A. Create:
- `core/uncertainty.py`

Add:
1. `fit_residual_conformal_calibrator(y_true, y_pred, alpha=0.1) -> dict`
2. `apply_conformal_interval(y_pred, calibrator) -> pd.DataFrame`
3. `evaluate_interval_quality(y_true, lower, upper) -> dict`
Metrics:
- coverage
- mean interval width
- median interval width

B. Integrate uncertainty into the benchmark flow:
- optional calibration split inside each training fold
- compute coverage/width for RS, Thickness, RSU where possible

C. Update schemas:
- store uncertainty metrics in fold results and benchmark summaries

D. Update Model tab benchmark UI:
- add optional uncertainty columns
- add interval-quality section

E. Add tests:
- `tests/test_uncertainty.py`
Test:
1. conformal intervals have valid lower/upper ordering
2. interval metrics compute correctly
3. benchmark can include uncertainty metrics when enabled

Design rules:
- Keep uncertainty optional
- Do not break existing point-prediction workflows
- Make interval evaluation available to all future models through the same interface

Output:
- Add model-agnostic uncertainty calibration
- Add uncertainty metrics to benchmark results
- Add tests