Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a model-processing visualization workflow so the user can see how raw data becomes training data and recommendation inputs.

Goal:
Make the modeling pipeline transparent by visualizing how data is processed step by step.

Implement the following:
1. Add a new subsection in the model tab called “Data Processing View” or similar
2. Show the following stages for the currently selected material and target instance:
   - raw imported rows
   - cleaned / normalized rows
   - grouped target instances
   - derived columns added
   - in-spec / out-of-spec labeling
   - rows selected for baseline trend fitting
   - rows selected for model training
   - rows excluded and why
3. Add tables and/or summaries for each stage:
   - row count
   - missing values count
   - in-spec count
   - target-instance count
   - feature columns used
4. Add a pipeline diagram or sequential cards that visually show:
   raw -> normalized -> labeled -> feature-engineered -> trained model -> recommendation candidates
5. Allow the user to click each stage to inspect a preview dataframe
6. Add export support for these intermediate tables if practical

Implementation guidance:
- Put core logic in a reusable pipeline-inspection module
- Keep UI read-only for this feature
- Make sure different materials are not mixed
- Respect target-instance grouping

Output:
- Add a processing-visualization section
- Show intermediate dataset states and counts
- Summarize how data flows through the model