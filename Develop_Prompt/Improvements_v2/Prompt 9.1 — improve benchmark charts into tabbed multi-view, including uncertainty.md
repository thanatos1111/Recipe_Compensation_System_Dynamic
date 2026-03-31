Refactor the Benchmark chart UI in `Recipe_Compensation_System_Dynamic` from a single metric dropdown into chart tabs, and add uncertainty chart views.

Goal:
Make benchmark visualization easier to read and compare across multiple metrics without repeatedly changing one dropdown.

Implement the following:

A. Update:
- `ui/model_panel.py`

B. Remove the chart metric selector from the Benchmark setup form:
- remove or deprecate `bench_chart_metric_combo` from the setup area

C. In the chart area, replace the single chart control flow with chart tabs:
1. `Spec-pass`
2. `RS`
3. `Thickness`
4. `RSU`
5. `Uncertainty`
6. `Backtest` (placeholder or active only if backtest exists)

D. Chart behavior:
1. `Spec-pass`
   - show spec-pass accuracy comparison across bundles
2. `RS`
   - show RS MAE comparison
3. `Thickness`
   - show Thickness MAE comparison
4. `RSU`
   - show RSU MAE comparison
5. `Uncertainty`
   - if uncertainty enabled and available:
     - show coverage and width comparisons
   - otherwise show readable placeholder text
6. `Backtest`
   - if recommendation backtest exists:
     - show backtest comparison summary
   - otherwise show readable placeholder text

E. Requirements:
1. Keep existing summary/fold tables unchanged.
2. Use the same benchmark results already stored in the panel.
3. The chart area should update when benchmark results change.
4. The selected ranking objective should not change chart availability.
5. Keep code modular enough for future auto-decision engine reuse.

F. Add helper methods as needed:
- `_render_benchmark_chart_spec_pass(...)`
- `_render_benchmark_chart_rs(...)`
- `_render_benchmark_chart_thickness(...)`
- `_render_benchmark_chart_rsu(...)`
- `_render_benchmark_chart_uncertainty(...)`
- `_render_benchmark_chart_backtest(...)`

G. Add tests only for helper formatting/state logic if practical.

Output:
- summarize chart tabs added
- explain how users now switch benchmark visualizations