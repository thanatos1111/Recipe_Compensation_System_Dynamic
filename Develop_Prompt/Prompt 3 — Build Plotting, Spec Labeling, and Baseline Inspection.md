Implement Milestone 3 for the `recipe_comp_system` project: plotting, spec labeling, and baseline inspection workflow.

Scope:
- For a selected material and optional selected target instance.
- Add spec configuration logic.
- Label each run as in-spec or out-of-spec.
- Show per-material and per-target-instance plots for inspection.
- Do not build predictive models yet.

Implement the following:

1. In `core/labeling.py`:
   - `compute_derived_features(df, spec_config)`
   - `apply_spec_labels(df, spec_config)`
2. Derived fields should include where applicable:
   - in_spec
   - rs_error
   - abs_rs_error
   - rsu_margin
   - thickness_error
   - total_flow
   - o2_ratio
   - delta values compared to previous run within the same target instance by lifetime order
3. Support both target mode and range mode:
   - RS target or RS min/max
   - thickness target or thickness min/max
   - RSU max threshold
4. In `ui/spec_config_panel.py`:
   - create a working spec settings panel
   - let the user set RS target or range
   - thickness target or range
   - RSU threshold
   - apply settings to currently selected material
5. In `ui/trend_panel.py`:
   - create plots for selected material and target instance:
     - parameters vs lifetime
     - RS vs lifetime
     - thickness vs lifetime
     - RSU vs lifetime
   - visually distinguish in-spec vs out-of-spec points
   - support switching between key parameters
   - support overlay comparison across target instances of the same material
6. In `ui/raw_table_panel.py`:
   - display labeled dataframe columns
   - allow filtering by all / in-spec / out-of-spec
   - allow filtering by target instance
7. In `core/trend_fitting.py`:
   - implement baseline in-spec trend extraction for one parameter vs lifetime
   - default method should be windowed median or piecewise linear, not high-order polynomial
   - include quantization to parameter step size
   - keep implementation simple and robust
8. Add a simple baseline compensation table generator by lifetime bin
9. Add tests for:
   - label logic
   - derived columns
   - in-spec filtering
   - trend fitting and quantization

Important design rules:
- Do not mix data across different materials.
- Same-material target instances may be compared but must remain identifiable.
- Keep plotting code separate from labeling logic.
- Make baseline trend output interpretable and stepwise.
- Polynomial can be optional comparison later, but not default.

Output:
- Working labeling and plots.
- Baseline compensation reference tables for inspection.
- Clear summary of files changed and how the feature works.