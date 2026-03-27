Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a dynamic visualization showing selected parameters versus lifetime, including both historical data and model-predicted values from the current in-state model.

Goal:
The user should be able to step through records over time and see what the model would have predicted using only data available up to that point.

Implement the following:
1. Add a dynamic analysis view for a selected material and target instance
2. Let the user choose several parameters / outputs of interest, such as:
   - linear offset
   - incident angle
   - Ar / O2
   - RS
   - Thickness
   - RSU
3. Show historical actual data vs lifetime
4. Add a stepping / playback mode where the user moves through each record in order
5. For each step:
   - train or simulate the model using only data from the beginning up to that record
   - show what the model would predict for the next point or current context
6. Overlay:
   - historical actuals
   - model-predicted trajectory
   - current recommendation if applicable
7. Add controls:
   - next / previous record
   - autoplay
   - choose outputs to display
   - choose whether to retrain at each step or use cached staged models

Implementation guidance:
- This is primarily an analysis/visualization tool
- Optimize for clarity first, performance second
- Cache staged results if needed
- Keep same-material data logic correct and do not leak future rows into historical-step prediction

Output:
- Add dynamic step-through visualization of historical vs model state
- Make staged model behavior visible across the lifetime