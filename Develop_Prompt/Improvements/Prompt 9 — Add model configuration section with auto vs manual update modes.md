Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a model configuration section so the user can inspect and adjust model behavior, including auto-update versus manual-update modes.

Implement the following:
1. Add a “Model Configuration” section in the app
2. Show current model-related settings such as:
   - model type
   - fallback rules
   - training window / included target instances
   - optimizer weights
   - safe-band settings
   - update mode
3. Support at least two update modes:
   - automatic adjustment based on follow-up data
   - manual adjustment / manual retrain control
4. Let the user configure:
   - retrain trigger behavior
   - confidence thresholds
   - candidate search width
   - whether current target data should get higher weight
   - whether machine-event segmentation is enabled if available
5. Add validation and persistence of these settings
6. Make sure changes are reflected in live model/recommendation behavior where appropriate

Implementation guidance:
- Keep settings centrally managed
- Separate runtime config from UI widgets
- Avoid exposing raw internal objects; present engineering-friendly controls

Output:
- Add model configuration UI and backing config logic
- Support auto/manual operation modes