# Recipe Parameter Compensation System
## Cursor-Ready Technical Design Spec v2

Version: 2.0  
Scope: Demonstration app and algorithm framework for material-based, target-instance-aware process recipe compensation in plasma deposition

---

# 1. Purpose

Build a software system that ingests historical deposition records from an Excel workbook and produces a **material-specific, target-instance-aware, updateable recipe compensation workflow**.

The system must help users maintain deposition output within specification across the lifetime of each physical target by recommending recipe adjustments based on:
- historical runs from previous targets of the **same material**,
- the current active target’s own run history,
- current machine situation or recent drift indicators when available.

The deposited film may vary in:
- thickness,
- sheet resistance (`RS`),
- sheet resistance uniformity (`RSU`).

The compensation goal is to recommend recipe settings that maximize the probability of achieving:
- `RSU` below a threshold,
- `RS` close to a target or within a target range,
- thickness close to a target or within a target range,
across the usable lifetime of a target.

The system is **not** just a polynomial curve fitter. It is a layered compensation framework that:
1. imports and cleans historical data,
2. groups runs by material and target instance,
3. labels in-spec and out-of-spec runs,
4. learns response of film outputs to recipe, lifetime, and context,
5. generates stepwise constrained recipe recommendations,
6. updates recommendations as new runs arrive.

---

# 2. Core Domain Hierarchy

The correct process hierarchy is:

`Workbook -> Material -> Target Instance -> Deposition Runs`

## 2.1 Meaning of each level

### Workbook
One Excel workbook may contain multiple material sheets.

### Material
Each sheet represents one **material**.
Examples:
- `Ta`
- `Ru`
- `Pt`

Different materials must be analyzed independently.

### Target Instance
For one material, there can be multiple physical targets used sequentially over time.
When one target is consumed and replaced by a new one of the same material, that new target is a **new target instance**.

Each target instance has:
- its own `Target ID`
- its own lifetime starting from zero
- its own run history

Examples for material `Ta`:
- `Ta.1`
- `Ta.2`
- `Ta.3`

### Deposition Runs
Each row in the raw sheet represents one deposition run or one deposition-result record associated with:
- a material,
- a target instance,
- a lifetime,
- recipe settings,
- measured outputs.

---

# 3. Key User Constraints and Assumptions

## 3.1 Excel input format
- Input file is a single `.xlsx` workbook.
- First row of each sheet contains column names.
- Each sheet corresponds to one **material**.
- The sheet name is the **material name**.
- Different materials are analyzed independently.
- Within one material sheet, rows may belong to multiple target instances of the same material.
- Target identity is explicitly given by the `Target ID` column.
- The importer should group rows by **sheet name + Target ID**.
- Lifetime reset inference is not the primary grouping method when `Target ID` exists.

## 3.2 Required raw columns and order
For each material sheet, the expected columns from left to right are:
1. `LOT ID`
2. `Date`
3. `Wafer ID`
4. `KW.H`
5. `Ar`
6. `O2`
7. `Incident Angle`
8. `Linear Offset`
9. `Rotations`
10. `Thickness`
11. `RS`
12. `RSU`
13. `Target ID`

## 3.3 Field meanings
- `LOT ID`: lot identifier
- `Date`: deposition or measurement date
- `Wafer ID`: wafer identifier
- `KW.H`: target lifetime in **kW*hour**
- `Ar`: Ar gas flow
- `O2`: O2 gas flow
- `Incident Angle`: recipe parameter
- `Linear Offset`: recipe parameter
- `Rotations`: recipe parameter
- `Thickness`: measured film thickness
- `RS`: measured sheet resistance
- `RSU`: measured sheet resistance uniformity
- `Target ID`: identifier for the physical target instance

## 3.4 Target ID rule
`Target ID` is mandatory in the standard input format.

Recommended format:
- `[material].[number]`

Examples:
- `Ta.1`
- `Ta.2`
- `Ru.1`

Interpretation:
- material = `Ta`
- target instance = `1`, `2`, etc.

The system should validate that the `Target ID` is consistent with the material sheet name where applicable.

## 3.5 Compensation goals
The system should recommend recipe settings that aim to satisfy:
- `RSU` below a threshold,
- `RS` close to a target or within a range,
- thickness close to a target or within a range,
- stable and repeatable performance across target lifetime.

## 3.6 Material and target handling rule
- Do **not** mix different materials.
- Do allow multiple target instances of the **same material** to contribute historical knowledge.
- Keep target-instance identity explicit.
- Recommendations are generated for the **currently active target instance** of the selected material.

---

# 4. Product Goals

## 4.1 Phase goals

### Phase 1
Data import, cleaning, grouping by material and target instance, visualization, in-spec filtering, baseline compensation display.

### Phase 2
Material-specific predictive models for `RS`, `Thickness`, and `RSU`, with target-instance-aware handling.

### Phase 3
Discrete constrained compensation engine that proposes next recipe settings for the current target instance.

### Phase 4
Target-instance update and machine-drift-aware correction using new runs.

### Phase 5
Deployment-oriented workflow, confidence scoring, review tools, and operator guidance.

---

# 5. Functional Requirements

## 5.1 Import
- Load one Excel workbook.
- Read all material sheets.
- Assign `material_name` from sheet name.
- Validate required columns.
- Normalize column names and units.
- Group rows inside each material sheet by `Target ID`.
- Save imported data into structured storage.

## 5.2 Inspection and analysis
For a selected material, user must be able to:
- view all raw records,
- inspect target instances under that material,
- define spec rules,
- mark in-spec / out-of-spec,
- plot parameters vs lifetime,
- plot `RS`, `Thickness`, `RSU` vs lifetime,
- compare target instances of the same material,
- inspect parameter step history,
- inspect compensation suggestion table.

## 5.3 Compensation modeling
For a selected material and active target instance, system must:
- estimate baseline compensation trends,
- learn relationships between recipe parameters and outputs,
- include out-of-spec runs in response modeling,
- use same-material historical target instances as prior data,
- prioritize current target instance data,
- generate recommended recipe candidates,
- enforce parameter discreteness and constraints,
- show confidence and predicted outcomes.

## 5.4 Updating
For the active target instance, system must:
- accept new runs with recipe + measured outputs,
- append new runs to the correct target instance,
- refresh material-level and instance-level artifacts,
- update compensation recommendations,
- preserve prior model versions or snapshots if needed.

---

# 6. Non-Functional Requirements

- Clear and explainable outputs.
- Isolation across different materials.
- Explicit target-instance traceability.
- Deterministic reproducibility with fixed random seeds.
- Configurable spec thresholds and optimizer weights.
- Modular architecture for future embedding into machine software.
- Ability to export recipe tables, plots, and reports.

---

# 7. High-Level Algorithm Strategy

The system uses a five-layer approach.

## Layer A — Data & labeling
Convert raw material sheets into standardized records with explicit material and target-instance grouping.

## Layer B — Baseline trend extraction
Use in-spec runs to estimate stepwise or piecewise compensation references versus lifetime.
This can be generated per target instance and summarized at material level.

## Layer C — Material-level response modeling
Use same-material historical data from multiple target instances to learn common relationships between:
- lifetime,
- recipe,
- outputs.

## Layer D — Target-instance correction
Adjust the material-level model using data from the current active target instance and recent machine context.

## Layer E — Recommendation engine
Given current lifetime and desired targets, generate valid candidate recipes, predict outcomes, score them, and select the best next recommendation.

---

# 8. Why Material-Level Modeling With Target Boundaries Is Required

The earlier simplified assumption of “one sheet = one independent target” is not sufficient.

Correct production logic:
- different materials should remain isolated,
- same-material target instances should be available as historical support,
- the current target instance should remain identifiable and separately inspectable,
- the recommendation engine should not flatten away target-instance identity.

Implications:
- data becomes richer for the same material,
- early-life gaps for a current target can be partially supported by prior targets of the same material,
- target drift and target-specific deviation still need explicit correction,
- UI and models must preserve both material and target-instance structure.

---

# 9. Data Model

## 9.1 Internal entities

### MaterialDataset
Represents one material sheet plus grouped target instances.

Fields:
- `material_name: str`
- `source_sheet_name: str`
- `all_records: DataFrame`
- `target_instances: dict[str, TargetInstance]`
- `column_map: dict`
- `spec_config: SpecConfig`
- `material_model_artifacts: MaterialModelArtifacts`
- `recommendation_artifacts: RecommendationArtifacts | None`
- `warnings: list[str]`

### TargetInstance
Represents one physical target lifecycle under a material.

Fields:
- `target_id: str`
- `material_name: str`
- `records: DataFrame`
- `status: str`  
  Suggested values: `historical`, `active`, `retired`
- `fit_artifacts: FitArtifacts | None`
- `instance_correction_artifacts: InstanceCorrectionArtifacts | None`
- `warnings: list[str]`

### SpecConfig
Fields:
- `rs_target: float | None`
- `rs_min: float | None`
- `rs_max: float | None`
- `rs_tol: float | None`
- `thickness_target: float | None`
- `thickness_min: float | None`
- `thickness_max: float | None`
- `thickness_tol: float | None`
- `rsu_max: float`
- `use_rs_target_mode: bool`
- `use_thickness_target_mode: bool`

### ParameterConfig
For each recipe parameter:
- `name`
- `type` (`continuous`, `integer`, `categorical`, `boolean`)
- `min_value`
- `max_value`
- `step`
- `allowed_values`
- `is_enabled`
- `is_coupled`
- `coupling_group`

### MaterialModelArtifacts
Fields:
- `rs_model`
- `thickness_model`
- `rsu_model`
- `spec_classifier` (optional)
- `feature_schema`
- `metrics`
- `train_summary`
- `confidence_summary`

### InstanceCorrectionArtifacts
Fields:
- `bias_terms`
- `recent_residual_summary`
- `instance_confidence`
- `drift_summary`

### FitArtifacts
Fields:
- `parameter_trend_fits`
- `lifetime_bins`
- `in_spec_region_maps`
- `stepwise_reference_table`

### RecommendationArtifacts
Fields:
- `candidate_table`
- `recommended_recipe`
- `predicted_outputs`
- `score_breakdown`
- `safe_band`
- `confidence_summary`

---

# 10. Data Standardization and Derived Fields

Each imported row should be transformed into a canonical structure.

## 10.1 Canonical columns
Required canonical columns:
- `lot_id`
- `date`
- `wafer_id`
- `material_name`
- `target_id`
- `lifetime`
- `incident_angle`
- `linear_offset`
- `rotations`
- `ar_flow`
- `o2_flow`
- `thickness`
- `rs`
- `rsu`

Default raw-to-canonical mapping:
- `LOT ID` -> `lot_id`
- `Date` -> `date`
- `Wafer ID` -> `wafer_id`
- `KW.H` -> `lifetime`
- `Ar` -> `ar_flow`
- `O2` -> `o2_flow`
- `Incident Angle` -> `incident_angle`
- `Linear Offset` -> `linear_offset`
- `Rotations` -> `rotations`
- `Thickness` -> `thickness`
- `RS` -> `rs`
- `RSU` -> `rsu`
- `Target ID` -> `target_id`

`material_name` should normally be taken from the sheet name.

Optional columns for future extension:
- `run_id`
- `operator`
- `machine_id`
- `maintenance_state`
- `notes`

## 10.2 Derived columns
Add these derived fields:
- `in_spec`
- `rs_error = rs - rs_target` if target mode is enabled
- `abs_rs_error = abs(rs - rs_target)`
- `rsu_margin = rsu_max - rsu`
- `thickness_error = thickness - thickness_target` if target mode enabled
- `total_flow = ar_flow + o2_flow`
- `o2_ratio = o2_flow / total_flow` when total_flow > 0
- `run_index_within_target_sorted_by_lifetime`
- `delta_<parameter>` compared with previous run within the same target instance

## 10.3 In-spec label logic
A run is in spec if all enabled criteria pass.

Example logic:
- `rsu <= rsu_max`
- `rs_min <= rs <= rs_max` or `abs(rs - rs_target) <= rs_tol`
- `thickness_min <= thickness <= thickness_max` or `abs(thickness - thickness_target) <= thickness_tol`

Implement this in a reusable labeling function.

---

# 11. Data Validation Rules

The importer must validate per material sheet and per target instance:
- required fields are present,
- `Target ID` column exists,
- numeric columns parse correctly,
- no impossible negative values where forbidden,
- discrete parameter values respect declared step or are at least warnable,
- if total flow is constrained, detect whether `ar_flow + o2_flow` is approximately constant,
- `Target ID` values are not blank for usable records,
- `Target ID` prefix is consistent with the sheet/material name where expected, for example `Ta.1` inside sheet `Ta`.

Warnings should include:
- missing required columns,
- missing `Target ID`,
- malformed `Target ID`,
- low record count,
- missing early lifetime region,
- large gaps in lifetime coverage,
- duplicated lifetime rows within the same target instance,
- suspicious outliers.

---

# 12. UI / Workflow Design

## 12.1 Main workflow
1. Open workbook
2. Parse sheets into material datasets
3. Select a material
4. Inspect target instances under that material
5. Mark one target instance as active/current
6. Configure spec and parameter constraints
7. Inspect plots and historical behavior
8. Train / refresh material-level and instance-level artifacts
9. Request recommendation for a chosen lifetime / current state
10. Review candidate table and selected recipe
11. Add new run results later and update target analysis

## 12.2 Main panels

### A. Workbook / material navigator
- workbook path
- material sheet list
- material summary stats
- warnings per material

### B. Target instance selector
- target instance list for selected material
- active/current target marker
- per-instance summary stats

### C. Raw data panel
- sortable table
- filters for all / in-spec / out-of-spec
- filters by target instance
- export option

### D. Spec configuration panel
- `RS` target/range settings
- thickness target/range settings
- `RSU` threshold
- apply / relabel button

### E. Trend analysis panel
- parameter vs lifetime
- `RS` vs lifetime
- thickness vs lifetime
- `RSU` vs lifetime
- in-spec highlighting
- fitted trend overlays
- per-target-instance overlays within the same material

### F. Model panel
- material-level training summary
- instance correction summary
- model type selection
- metrics
- confidence notes

### G. Recommendation panel
- current lifetime input
- current recipe input or latest recipe load
- candidate generation settings
- recommended recipe table
- predicted `RS` / thickness / `RSU`
- safe band / feasible region display
- explanation / score breakdown

### H. Update panel
- append new run manually or import from file
- assign to target instance
- retrain / update button
- version snapshot notes

---

# 13. Phase-by-Phase Technical Plan

# Phase 1 — Ingestion, Labeling, Visualization, Baseline Trend Extraction

## Objective
Build a usable inspection and analysis application that supports:
- material-level import,
- target-instance grouping,
- spec labeling,
- baseline trend views.

## Deliverables
- Excel workbook importer
- per-sheet material parser
- target-instance grouping by `Target ID`
- configurable column mapping
- spec config and in-spec labeling
- raw data table
- main trend plots
- baseline trend fitting module

## 13.1 Algorithms in Phase 1

### 13.1.1 Baseline compensation trend
For each adjustable parameter and selected target instance or material summary view:
1. sort rows by lifetime within the target instance,
2. keep only in-spec rows,
3. fit trend against lifetime,
4. convert trend into stepwise valid values.

Recommended fitting priority:
1. piecewise linear fit
2. spline with smoothing and then quantization
3. simple moving median by lifetime window
4. polynomial only as optional comparison, not default

### 13.1.2 Why not default polynomial
- unstable at edges,
- poor behavior with sparse data,
- may oscillate,
- less suitable for discrete step conversion.

### 13.1.3 Default baseline method
Use **windowed median or piecewise linear fit + quantization** as default.

Reason:
- robust to noise,
- visually interpretable,
- close to operator tuning style,
- easier to explain,
- consistent with stepwise recipe settings.

## 13.2 Outputs of Phase 1
- parameter history plots
- in-spec-only parameter trend plots
- baseline compensation tables per lifetime bin
- comparison across target instances of the same material
- warnings where data is insufficient

## 13.3 Lifetime binning
To stabilize sparse data, support lifetime binning.

Example:
- equal-width bins,
- bins by run count,
- user-configurable lifetime interval.

For each bin, compute:
- count
- median parameter values
- `RS` median
- thickness median
- `RSU` median
- in-spec fraction

These tables can become the first version of a compensation reference.

---

# Phase 2 — Material-Specific Response Modeling With Target-Instance Awareness

## Objective
For each material independently, model how outputs respond to recipe and lifetime while preserving target-instance identity.

## Deliverables
- material-level `RS` model
- material-level thickness model
- material-level `RSU` model
- optional in-spec classifier
- instance-level correction summary
- metrics and confidence report

## 14.1 Training data
Use **all rows** from the selected material, including out-of-spec runs.

Required features:
- lifetime
- incident_angle
- linear_offset
- rotations
- ar_flow and/or o2_flow / o2_ratio
- total_flow if not fixed
- target_id encoded or represented by grouped correction strategy
- optional previous-run deltas within target instance
- optional machine-state indicators when available

Targets:
- `rs`
- `thickness`
- `rsu`
- optional `in_spec`

## 14.2 Model choices
Default order of preference:
1. Gradient Boosting Regressor
2. Random Forest Regressor
3. GAM / spline regression if interpretability is prioritized
4. Linear / polynomial baseline for comparison only

Because some materials may still have limited data, the app should support model fallback rules.

### Fallback strategy
- if sample size is very small, use simpler models,
- if sample size is moderate, enable tree-based models,
- if lifetime coverage is sparse, report low confidence.

## 14.3 Instance correction layer
For the active target instance:
- compare actual outputs to material-model predictions,
- estimate recent bias / residual trend,
- adjust future predictions using this instance-specific correction.

This lets the system:
- use same-material historical knowledge,
- adapt to the current target instance.

## 14.4 Validation strategy
Because data is sequential in lifetime, use time-aware validation where possible.

Recommended:
- sort by date and/or lifetime within target instances,
- use forward split or blocked cross-validation,
- avoid relying only on random split metrics.

Metrics:
- MAE for `RS`
- MAE for thickness
- MAE for `RSU`
- optionally RMSE
- classification AUC / accuracy if classifier enabled

## 14.5 Model interpretation outputs
For user transparency, provide:
- feature importance
- partial dependence plots or local sensitivity charts
- prediction residual plots vs lifetime
- instance-vs-material residual comparisons

---

# Phase 3 — Discrete Constrained Compensation Engine

## Objective
Generate recommended next recipe settings for the currently active target instance.

## Deliverables
- candidate generation engine
- constraint checker
- scoring function
- ranked recommendation list
- predicted output table

## 15.1 Recommendation inputs
For selected material and active target instance:
- current lifetime
- desired `RS` target or range
- desired thickness target or range
- `RSU` max
- current recipe or baseline recipe
- parameter constraints and step sizes
- optional recent machine-state context

## 15.2 Candidate generation
Generate valid candidate recipes around a reference point.

Reference point priority:
1. latest actual recipe on the active target instance
2. baseline compensation recipe at current lifetime
3. user-entered manual recipe

Candidate generation methods:
- local grid search,
- bounded neighborhood search,
- user-defined tuning-order search.

Each parameter must obey:
- min/max bounds,
- discrete step size,
- categorical allowed values,
- coupling rules.

## 15.3 Coupling rule example
If total gas flow must remain 60 sccm:
- use `o2_flow` as one search variable,
- compute `ar_flow = 60 - o2_flow`,
- reject invalid candidates where any flow is outside allowed bounds.

## 15.4 Scoring function
Each candidate receives predicted outputs and a score.

Suggested score:

`score = w_rs * normalized_rs_error`
`      + w_thickness * normalized_thickness_error`
`      + w_rsu * rsu_penalty`
`      + w_move * move_penalty`
`      - w_margin * safe_margin_reward`

Definitions:
- `normalized_rs_error = abs(rs_pred - rs_target) / rs_tol`
- `normalized_thickness_error = abs(t_pred - t_target) / thickness_tol`
- `rsu_penalty = max(0, rsu_pred - rsu_max) / rsu_max`
- `move_penalty` penalizes too many or too large adjustments from reference recipe
- `safe_margin_reward` rewards distance from spec boundary

Hard constraints should reject candidates before scoring.

## 15.5 Recommendation outputs
For top candidates show:
- full recipe values
- predicted `RS`
- predicted thickness
- predicted `RSU`
- in-spec probability if available
- score breakdown
- change vs current recipe
- material-model contribution vs instance-correction contribution when useful

---

# Phase 4 — Feasible Region and Safe Band Estimation

## Objective
Show not just one recipe line but an acceptable region or safe band around it.

## Deliverables
- safe range estimation per parameter
- feasible-region plots for major parameters
- confidence-aware recommendation band

## 16.1 Why safe band matters
A single fitted line is too rigid. Real operation often has a region of valid values. Users need to know:
- best center recommendation,
- acceptable adjustment space,
- risk when deviating from center.

## 16.2 Safe band estimation methods
Priority order:
1. candidate feasibility map from response models
2. margin-based thresholding
3. quantile prediction if added later

### Method A — Feasibility search
For selected lifetime:
- vary one parameter around center recommendation while respecting constraints,
- keep other parameters fixed or controlled,
- evaluate predictions,
- identify valid interval where specs are satisfied.

### Method B — Margin map
Compute:
- `rs_margin`
- `thickness_margin`
- `rsu_margin`

Safe band is where all margins are positive.

## 16.3 UI output
- center value marker
- lower / upper acceptable bounds
- confidence shading
- warning when band is narrow or unsupported by data

---

# Phase 5 — Incremental Update and Refit Workflow

## Objective
Update material-level and target-instance-level compensation artifacts as new runs are added.

## Deliverables
- append-run interface
- retraining pipeline
- change tracking
- model versioning or snapshot support

## 17.1 Update modes
### Mode A — full refit
For demo app, simplest and safest:
- append new row,
- relabel in-spec,
- refresh target-instance grouping,
- refit trend tables,
- retrain material-level models,
- refresh instance correction artifacts,
- regenerate recommendation artifacts.

### Mode B — light incremental update
Optional later:
- update derived tables,
- retrain only if enough new rows are added,
- otherwise update cached recommendations and instance correction only.

## 17.2 Retrain triggers
- manual user action,
- after every new run,
- after N new runs,
- after new lifetime segment appears,
- after maintenance or machine-state change.

For demonstration app, default to manual or semi-automatic retrain.

## 17.3 Model versioning
Store for each material and active target context:
- dataset snapshot timestamp
- model version
- config version
- training metrics
- optional rollback file

---

# 14. Detailed Algorithm Modules

## 18.1 Import module
Responsibilities:
- open workbook
- enumerate material sheets
- parse rows
- apply column mapping
- create `MaterialDataset`
- group records into `TargetInstance`s using `Target ID`

Proposed file:
- `core/importer.py`

Main functions:
- `load_workbook(path) -> dict[str, pd.DataFrame]`
- `normalize_material_sheet(df, column_map, config, material_name) -> pd.DataFrame`
- `build_material_dataset(sheet_name, df, ...) -> MaterialDataset`
- `group_target_instances(df) -> dict[str, TargetInstance]`

## 18.2 Labeling module
Responsibilities:
- apply spec config
- compute in-spec labels
- compute derived margins and errors

Proposed file:
- `core/labeling.py`

Functions:
- `apply_spec_labels(df, spec_config) -> pd.DataFrame`
- `compute_derived_features(df, spec_config) -> pd.DataFrame`

## 18.3 Trend fitting module
Responsibilities:
- build baseline compensation trends from in-spec data
- produce stepwise reference tables

Proposed file:
- `core/trend_fitting.py`

Functions:
- `fit_parameter_trend(df, parameter_name, method, step) -> FitResult`
- `build_lifetime_reference_table(df, config) -> pd.DataFrame`
- `quantize_series_to_step(values, step, min_val, max_val)`

## 18.4 Feature engineering module
Responsibilities:
- build material-level feature matrix
- preserve target-instance identity or grouped residual information

Proposed file:
- `core/feature_engineering.py`

Functions:
- `build_feature_matrix(df, feature_config)`
- `encode_target_identity(df, mode)`
- `build_instance_correction_inputs(df)`

## 18.5 Response model module
Responsibilities:
- train material-specific regression models
- evaluate performance
- expose prediction interface

Proposed file:
- `core/response_models.py`

Functions:
- `train_rs_model(df, feature_config)`
- `train_thickness_model(df, feature_config)`
- `train_rsu_model(df, feature_config)`
- `train_spec_classifier(df, feature_config)`
- `predict_outputs(model_bundle, X)`

## 18.6 Instance correction module
Responsibilities:
- estimate instance-specific correction for current active target
- summarize recent residual drift

Suggested file:
- `core/instance_correction.py`

Functions:
- `fit_instance_bias(material_model_bundle, instance_df)`
- `apply_instance_correction(predictions, correction_artifacts)`
- `summarize_recent_residuals(instance_df, material_model_bundle)`

## 18.7 Candidate generator
Responsibilities:
- create valid discrete candidate recipes
- enforce bounds and coupling

Proposed file:
- `core/candidate_generator.py`

Functions:
- `generate_candidates(reference_recipe, parameter_config, neighborhood_config)`
- `apply_coupling_rules(candidate, coupling_config)`
- `is_valid_candidate(candidate, parameter_config)`

## 18.8 Optimizer module
Responsibilities:
- predict candidate outcomes
- compute scores
- rank recommendations

Proposed file:
- `core/optimizer.py`

Functions:
- `score_candidate(candidate, predicted_outputs, spec_config, weights, reference_recipe)`
- `rank_candidates(candidates, model_bundle, context, ...)`
- `select_best_candidate(ranked_df)`

## 18.9 Safe band module
Responsibilities:
- estimate feasible bounds around center recommendation
- provide band tables for visualization

Proposed file:
- `core/safe_band.py`

Functions:
- `estimate_parameter_band(center_recipe, model_bundle, parameter_name, ...)`
- `build_feasible_map(center_recipe, varying_parameters, ...)`

## 18.10 Update module
Responsibilities:
- append new run
- refresh material dataset and artifacts

Proposed file:
- `core/update_pipeline.py`

Functions:
- `append_run_to_target_instance(material_dataset, target_id, new_row)`
- `refresh_material_artifacts(material_dataset, train_config)`
- `refresh_instance_artifacts(material_dataset, target_id, train_config)`

---

# 15. Data Storage Design

## 19.1 Recommended internal storage for demo app
Use:
- in-memory pandas structures during runtime,
- optional local project folder for saved configs and artifacts,
- optional SQLite in later phase.

## 19.2 File outputs
Per material and/or target instance, optionally save:
- cleaned CSV
- labeled CSV
- reference compensation table CSV
- trained model pickle/joblib
- plots PNG
- recommendation reports CSV
- config JSON

Suggested folder structure:

```text
project_root/
  data/
    source_workbook.xlsx
    cleaned/
      Ta_cleaned.csv
      Ru_cleaned.csv
  artifacts/
    Ta/
      material_model_bundle.pkl
      metrics.json
      config_snapshot.json
      Ta.1/
        reference_table.csv
        instance_correction.json
      Ta.2/
        reference_table.csv
        instance_correction.json
    Ru/
      ...
  app/
  core/
  ui/
  tests/
```

---

# 16. Model Selection Rules for Sparse Data

The app must expose simple model selection logic.

## 20.1 Suggested thresholds
Example only; make configurable.

At material level:
- `< 10 rows`: no predictive model, only descriptive plots and baseline medians
- `10–25 rows`: simple regression or piecewise fit only
- `25–60 rows`: tree models allowed with caution
- `> 60 rows`: full material-specific predictive workflow

At active target-instance level:
- `< 5 rows`: correction disabled or minimal bias correction only
- `5–15 rows`: light bias correction
- `> 15 rows`: residual-trend-based instance correction can be enabled

## 20.2 Confidence levels
For each material and active target instance, define:
- `Low confidence`
- `Medium confidence`
- `High confidence`

Based on:
- row count
- lifetime coverage
- in-spec count
- model validation metrics
- extrapolation amount from nearest known lifetime region
- amount of current target-instance data

---

# 17. Extrapolation Rules

Extrapolation is risky.

Rules:
- if requested lifetime is outside the supported historical range for the material or target instance, show warning,
- recommendation should prefer conservative small moves,
- confidence must be downgraded,
- optional clamp to nearest supported lifetime band.

---

# 18. Tuning-Policy Support

User noted that tuning style can differ. The first version should not try to learn universal operator policy across all situations.

Instead, add optional **tuning policy templates**.

## 21.1 Example tuning policy template
Allowed tuning order:
1. linear offset
2. incident angle
3. gas ratio
4. rotations
5. additional parameters when available

Use this in candidate generation to:
- prefer changing only early-priority parameters first,
- penalize lower-priority changes unless necessary.

## 21.2 Why this matters
This makes recommendations:
- more practical,
- more aligned with operating discipline,
- easier to review and accept.

---

# 19. Stepwise Compensation Representation

Do not store only smooth curves. Final compensation output must support stepwise recipe values.

## 22.1 Recommended representation
For each material and target instance lifetime bin, store:
- recommended recipe vector
- parameter-by-parameter values
- predicted outputs
- confidence score
- safe band bounds

Example table columns:
- `material_name`
- `target_id`
- `lifetime_bin_start`
- `lifetime_bin_end`
- `incident_angle_rec`
- `linear_offset_rec`
- `rotations_rec`
- `o2_flow_rec`
- `ar_flow_rec`
- `rs_pred`
- `thickness_pred`
- `rsu_pred`
- `confidence`

---

# 20. Pseudocode for Core Workflow

## 23.1 Import and material build

```python
for sheet_name, raw_df in workbook.items():
    material_df = normalize_material_sheet(raw_df, column_map, config, material_name=sheet_name)
    material_df = compute_derived_features(material_df, spec_config)
    material_df = apply_spec_labels(material_df, spec_config)
    target_instances = group_target_instances(material_df)

    material_dataset = MaterialDataset(
        material_name=sheet_name,
        source_sheet_name=sheet_name,
        all_records=material_df,
        target_instances=target_instances,
        ...
    )
    material_store[sheet_name] = material_dataset
```

## 23.2 Phase 1 baseline trend fitting

```python
def build_baseline_reference(instance_df, fit_config):
    df = instance_df.sort_values("lifetime")
    in_spec_df = df[df["in_spec"] == True]

    reference_table = []
    for parameter in adjustable_parameters:
        fit_result = fit_parameter_trend(
            in_spec_df,
            parameter_name=parameter.name,
            method=fit_config.method,
            step=parameter.step,
        )
        store_fit_result(parameter.name, fit_result)

    lifetime_bins = make_lifetime_bins(df, fit_config)
    for bin_range in lifetime_bins:
        bin_subset = in_spec_df[in_bin(in_spec_df, bin_range)]
        row = summarize_bin_to_reference_recipe(bin_subset, parameter_config)
        reference_table.append(row)

    return pd.DataFrame(reference_table)
```

## 23.3 Phase 2 material-model training

```python
def train_material_models(material_dataset, feature_config, model_config):
    df = material_dataset.all_records.sort_values(["target_id", "lifetime"])
    X = build_feature_matrix(df, feature_config)

    rs_model = fit_regression_model(X, df["rs"], model_config.rs_model)
    th_model = fit_regression_model(X, df["thickness"], model_config.thickness_model)
    rsu_model = fit_regression_model(X, df["rsu"], model_config.rsu_model)

    spec_model = None
    if model_config.enable_classifier:
        spec_model = fit_classifier(X, df["in_spec"], model_config.spec_model)

    return MaterialModelArtifacts(...)
```

## 23.4 Phase 2 instance correction

```python
def build_instance_correction(material_dataset, target_id):
    instance_df = material_dataset.target_instances[target_id].records.sort_values("lifetime")
    base_pred = predict_outputs(material_dataset.material_model_artifacts, instance_df)
    correction = fit_instance_bias(base_pred, instance_df)
    return correction
```

## 23.5 Phase 3 recommendation

```python
def recommend_recipe(material_dataset, active_target_id, current_context, weights):
    reference_recipe = get_reference_recipe(material_dataset, active_target_id, current_context)
    candidates = generate_candidates(
        reference_recipe,
        parameter_config=current_context.parameter_config,
        neighborhood_config=current_context.search_config,
    )

    correction = material_dataset.target_instances[active_target_id].instance_correction_artifacts

    ranked_rows = []
    for candidate in candidates:
        if not is_valid_candidate(candidate, current_context.parameter_config):
            continue

        candidate = apply_coupling_rules(candidate, current_context.coupling_config)
        pred = predict_outputs(material_dataset.material_model_artifacts, candidate, current_context)
        pred = apply_instance_correction(pred, correction)
        score = score_candidate(candidate, pred, current_context.spec_config, weights, reference_recipe)
        ranked_rows.append({**candidate, **pred, "score": score})

    ranked_df = pd.DataFrame(ranked_rows).sort_values("score")
    best = ranked_df.iloc[0]
    return ranked_df, best
```

## 23.6 Phase 5 update pipeline

```python
def update_with_new_run(material_dataset, target_id, new_row, config):
    instance_df = material_dataset.target_instances[target_id].records.copy()
    instance_df = pd.concat([instance_df, pd.DataFrame([new_row])], ignore_index=True)
    instance_df = compute_derived_features(instance_df, config.spec_config)
    instance_df = apply_spec_labels(instance_df, config.spec_config)
    material_dataset.target_instances[target_id].records = instance_df.sort_values("lifetime")

    material_dataset.all_records = rebuild_all_records(material_dataset)
    material_dataset.material_model_artifacts = train_material_models(material_dataset, config.feature_config, config.model_config)
    material_dataset.target_instances[target_id].instance_correction_artifacts = build_instance_correction(material_dataset, target_id)
    material_dataset.target_instances[target_id].fit_artifacts = build_baseline_reference(instance_df, config.fit_config)
    material_dataset.recommendation_artifacts = None
    return material_dataset
```

---

# 21. Proposed Project Structure for Cursor

```text
recipe_comp_system/
  app.py
  requirements.txt
  README.md
  core/
    importer.py
    schemas.py
    labeling.py
    validation.py
    trend_fitting.py
    feature_engineering.py
    response_models.py
    instance_correction.py
    candidate_generator.py
    constraints.py
    optimizer.py
    safe_band.py
    update_pipeline.py
    exports.py
  ui/
    main_window.py
    material_selector.py
    target_instance_selector.py
    raw_table_panel.py
    spec_config_panel.py
    trend_panel.py
    model_panel.py
    recommendation_panel.py
    update_panel.py
  config/
    default_config.json
  artifacts/
  tests/
    test_importer.py
    test_labeling.py
    test_trend_fitting.py
    test_constraints.py
    test_optimizer.py
```

---

# 22. Suggested Python Stack

For demo application:
- Python
- pandas
- numpy
- scikit-learn
- scipy
- openpyxl
- matplotlib
- optionally PySide6 for desktop UI

If UI is postponed, begin with notebook or script + minimal desktop shell.

---

# 23. Testing Plan

## 25.1 Unit tests
Test:
- Excel sheet parsing
- material sheet grouping
- `Target ID` grouping
- column mapping
- in-spec labeling logic
- step quantization
- gas coupling enforcement
- candidate validity
- scoring function

## 25.2 Integration tests
Test:
- full import of workbook with multiple material sheets
- material switching
- target-instance switching
- model training per material
- recommendation generation for active target instance
- append new run and refit

## 25.3 Edge-case tests
- material sheet with missing columns
- missing `Target ID`
- malformed `Target ID`
- target instance with only out-of-spec runs
- material with very few rows
- target with missing early life data
- fixed-flow coupling violated in raw data
- requested lifetime beyond known range

---

# 24. Phase Implementation Order for Cursor

## Milestone 1
- project skeleton
- workbook importer
- material dataset manager
- target-instance grouping
- spec config model
- raw table display

## Milestone 2
- plotting panel
- in-spec labeling and filtering
- lifetime bin summary table
- baseline compensation trend fitting

## Milestone 3
- material-specific predictive models
- instance correction summary
- metrics panel
- feature engineering module

## Milestone 4
- candidate generation
- constraints engine
- recommendation ranking
- top candidate display

## Milestone 5
- safe band estimation
- update pipeline
- export and snapshot support

## Milestone 6
- UX refinement
- warnings and confidence system
- tuning-policy template support

---

# 25. Development Notes for Cursor Prompts

When implementing in Cursor, enforce these design rules:
- Never mix datasets from different materials.
- Same-material target instances may be used together, but keep target boundaries explicit.
- Make material selection and active target-instance selection explicit in all UI and backend calls.
- Make spec configuration editable per material, with optional per-instance overrides later if needed.
- Prefer modular functions with testable pure logic in `core/`.
- Keep plotting and UI separate from model logic.
- Implement fallback logic for sparse data.
- Quantize all recommended numeric parameters to valid step size before output.
- Enforce gas coupling before final recommendation.
- Provide explainable score breakdowns.

---

# 26. Recommended First Build Scope

For the first working app, implement only the following:

1. Import workbook and parse material sheets.
2. Group rows by `Target ID` inside each material sheet.
3. Standardize columns and compute in-spec labels.
4. Show per-material and per-target-instance plots and baseline compensation tables.
5. Train simple material-specific models for `RS`, thickness, and `RSU`.
6. Apply light correction for the current active target instance.
7. Generate local discrete recipe candidates.
8. Rank and show recommended next recipe.
9. Allow appending new run rows and retraining.

This is the minimum complete vertical slice.

---

# 27. Final Design Position

This system should be implemented as a **material-specific, target-instance-aware adaptive compensation engine**.

It should not be framed as only “fit one parameter curve versus lifetime.”  
It should instead support:
- material-by-material analysis,
- multiple target lifecycles for the same material,
- explicit `Target ID` grouping,
- baseline in-spec compensation reference,
- all-data response modeling,
- current-target correction,
- discrete constrained recipe recommendation,
- safe band estimation,
- iterative update with new target data.

That structure gives a practical path from a demonstration tool to later machine-integrated compensation software.

---

# 28. Cursor Implementation Prompt Pack

Use the following prompts directly in Cursor. Each prompt is written so Cursor can implement one milestone cleanly without trying to solve the whole app at once. The prompts assume a Python desktop application structure, material-specific data handling, target-instance-aware grouping, and modular logic separation between `core/` and `ui/`.

General rules that apply to all prompts:
- Do not mix datasets from different materials.
- Each sheet is one material.
- Group rows inside a material by `Target ID`.
- Same-material target instances may be used together, but their identities must remain explicit.
- Keep code modular and testable.
- Do not hardcode column names; support configurable column mapping.
- Keep UI code separate from business logic.
- Prefer small, focused commits.
- Add docstrings and type hints.
- Do not introduce features beyond the milestone unless required to make the milestone work.

---

## Prompt 1 — Create Project Skeleton

```text
Build the initial project skeleton for a Python desktop application named `recipe_comp_system` based on the technical design spec already discussed.

Requirements:
1. Create the following top-level structure:
   - app.py
   - requirements.txt
   - README.md
   - core/
   - ui/
   - config/
   - artifacts/
   - tests/
2. Inside `core/`, create placeholder modules with docstrings and minimal class/function stubs:
   - importer.py
   - schemas.py
   - labeling.py
   - validation.py
   - trend_fitting.py
   - feature_engineering.py
   - response_models.py
   - instance_correction.py
   - candidate_generator.py
   - constraints.py
   - optimizer.py
   - safe_band.py
   - update_pipeline.py
   - exports.py
3. Inside `ui/`, create placeholder modules:
   - main_window.py
   - material_selector.py
   - target_instance_selector.py
   - raw_table_panel.py
   - spec_config_panel.py
   - trend_panel.py
   - model_panel.py
   - recommendation_panel.py
   - update_panel.py
4. Create a `config/default_config.json` with placeholder keys for:
   - column mapping
   - spec settings
   - parameter constraints
   - fit settings
   - model settings
   - optimizer weights
5. In `core/schemas.py`, define typed dataclasses or pydantic-style classes for:
   - SpecConfig
   - ParameterConfig
   - MaterialDataset
   - TargetInstance
   - MaterialModelArtifacts
   - InstanceCorrectionArtifacts
   - FitArtifacts
   - RecommendationArtifacts
6. Implement a minimal `app.py` entry point that starts the application shell, even if panels are placeholders.
7. Populate `README.md` with:
   - project purpose
   - current milestone scope
   - folder structure
   - how to run
8. Add basic tests that validate imports and schema object construction.

Technical constraints:
- Use Python.
- Use pandas/numpy/scikit-learn/openpyxl/matplotlib in requirements.
- If a desktop UI framework is needed, use PySide6.
- Keep the code clean and implementation-ready, not pseudo-code.
- Do not implement full logic yet; only create a robust scaffold.

Output:
- Create all files.
- Show a concise summary of what was added.
- Note any assumptions explicitly in comments or README.
```

---

## Prompt 2 — Build Excel Importer, Material Manager, and Target-Instance Grouping

```text
Implement Milestone 2 for the `recipe_comp_system` project: Excel importer, material manager, and target-instance grouping.

Scope:
- Read one `.xlsx` workbook.
- Treat each sheet as one material.
- Group rows inside each material sheet by `Target ID`.
- Build an in-memory material store keyed by material/sheet name.
- Build nested target-instance storage under each material.

Implement the following:

1. In `core/importer.py`:
   - `load_workbook(path: str) -> dict[str, pd.DataFrame]`
   - `normalize_material_sheet(df: pd.DataFrame, column_map: dict, config: dict, material_name: str) -> pd.DataFrame`
   - `group_target_instances(df: pd.DataFrame) -> dict[str, TargetInstance]`
   - `build_material_dataset(sheet_name: str, df: pd.DataFrame, ...) -> MaterialDataset`
2. In `core/validation.py`:
   - column presence validation
   - numeric parsing checks
   - warnings for low row count, duplicate lifetime within target instance, missing required columns, suspicious negative values, missing or malformed `Target ID`
3. In `core/schemas.py` if needed:
   - finalize `MaterialDataset` and `TargetInstance` so they store raw records, normalized records, grouped target instances, warnings, and config references
4. Create a material manager layer that:
   - stores imported materials by name
   - exposes methods to list materials
   - retrieve one material
   - replace one material
   - retrieve target instances under a material
5. In the UI:
   - add workbook open action
   - display material list in `material_selector.py`
   - display target instance list in `target_instance_selector.py`
   - allow user to click a material and then a target instance to load data into `raw_table_panel.py`
6. Add column mapping support:
   - use values from `config/default_config.json`
   - expected standard columns are:
     `LOT ID, Date, Wafer ID, KW.H, Ar, O2, Incident Angle, Linear Offset, Rotations, Thickness, RS, RSU, Target ID`
7. Keep materials isolated:
   - no shared training data across materials
   - only same-material target instances may be grouped together
8. Add tests:
   - workbook with multiple material sheets
   - correct material count
   - correct `Target ID` grouping
   - correct per-instance row loading
   - validation warning generation

Important design rules:
- First row contains parameter names.
- Sheet name is material name.
- `Target ID` is mandatory in standard input.
- Normalize to canonical columns such as material_name, target_id, lifetime, incident_angle, linear_offset, rotations, ar_flow, o2_flow, rs, thickness, rsu.
- If canonical mapping fails, keep warnings clear and non-fatal where possible.

Output:
- Implement working importer, material manager, and target-instance grouping.
- Update README with usage instructions.
- Summarize created classes/functions and assumptions.
```

---

## Prompt 3 — Build Plotting, Spec Labeling, and Baseline Inspection

```text
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
```

---

## Prompt 4 — Build Material-Specific Model Training and Instance Correction

```text
Implement Milestone 4 for the `recipe_comp_system` project: material-specific model training for RS, thickness, and RSU, plus current target-instance correction.

Scope:
- Models must be trained independently per selected material.
- Do not pool data across different materials.
- Use all rows from the selected material, including out-of-spec rows.
- Preserve target-instance identity.
- Provide metrics and confidence notes.

Implement the following:

1. In `core/feature_engineering.py`:
   - build feature matrix from canonical material dataframe
   - include lifetime, incident_angle, linear_offset, rotations, o2_flow or o2_ratio, total_flow if needed
   - support target-instance-aware features or grouped handling
   - ensure consistent feature schema storage
2. In `core/response_models.py`:
   - `train_rs_model(...)`
   - `train_thickness_model(...)`
   - `train_rsu_model(...)`
   - optional `train_spec_classifier(...)`
   - `predict_outputs(model_bundle, X)`
3. In `core/instance_correction.py`:
   - fit a light correction for the active target instance based on residuals from the material-level model
   - provide a function to apply that correction to predictions
4. Use practical default models:
   - start with Gradient Boosting Regressor or Random Forest Regressor
   - add fallback to simpler models if row count is too low
5. Validation:
   - use time-aware or grouped validation where practical
   - avoid only random split metrics
6. Store results in artifacts:
   - trained material models
   - feature schema
   - training metrics
   - row count summary
   - confidence summary
   - active-instance correction summary
7. In `ui/model_panel.py`:
   - add controls to train models for selected material
   - display key metrics for RS, thickness, and RSU
   - show active target-instance correction summary
   - show warnings for low-data conditions
8. Add tests for:
   - per-material training isolation
   - target-instance-aware feature handling
   - model artifact creation
   - basic prediction call
   - instance correction output

Important design rules:
- Never train on multiple materials together.
- Out-of-spec rows are allowed and useful for response modeling.
- Keep model training logic in `core/`, not UI.
- Expose low confidence clearly when data is sparse or lifetime coverage is poor.

Output:
- Working per-material model training pipeline.
- Active-instance correction workflow.
- Model metrics panel in UI.
- Summary of model choices, fallback logic, and assumptions.
```

---

## Prompt 5 — Build Optimizer and Recommendation UI

```text
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
```

---

## Optional Prompt 6 — Append New Runs and Refresh Material / Instance Artifacts

```text
Implement the next milestone for `recipe_comp_system`: append new run data for the selected active target instance and refresh all related artifacts.

Scope:
- Add new row(s) to one target instance only.
- Recompute derived fields and labels.
- Rebuild baseline trend artifacts.
- Retrain or refresh material-level models if needed.
- Refresh target-instance correction.
- Refresh recommendation state.

Implement:
1. `core/update_pipeline.py` with append and refresh functions.
2. `ui/update_panel.py` for manual row input or CSV append.
3. Target-instance-specific retrain trigger.
4. Optional artifact snapshot saving.
5. Tests for append + refresh behavior.

Keep it simple and deterministic. Do not introduce cross-material learning.
```

---

## Suggested Usage Order in Cursor

Use the prompts in this order:
1. Prompt 1 — project skeleton
2. Prompt 2 — importer, material manager, target-instance grouping
3. Prompt 3 — plotting and labeling
4. Prompt 4 — model training and instance correction
5. Prompt 5 — optimizer and recommendation UI
6. Optional Prompt 6 — update pipeline

Recommended workflow in Cursor for each prompt:
- paste one prompt at a time,
- let Cursor implement,
- review generated files,
- run tests,
- commit before moving to the next prompt.

---

## Suggested Guardrail Sentence to Add to Every Cursor Prompt

You can prepend this sentence to any of the milestone prompts if needed:

```text
Follow the existing project structure and technical design spec. Keep changes scoped strictly to this milestone. Do not refactor unrelated modules unless necessary to make the milestone work.
```

