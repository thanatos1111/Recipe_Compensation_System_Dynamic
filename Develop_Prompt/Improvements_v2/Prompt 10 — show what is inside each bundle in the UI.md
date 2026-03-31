Improve model-bundle transparency in the Benchmark UI for `Recipe_Compensation_System_Dynamic`.

Goal:
Make it obvious to the user what each benchmark bundle contains for RS / Thickness / RSU modeling.

Implement the following:

A. Update:
- `ui/model_panel.py`
- optionally add a small helper module if needed, such as `core/bundles.py`

B. Add a bundle details panel in the Benchmark section:
When a bundle is selected in the model-bundle list, show:
- bundle name
- source: preset or custom
- RS model
- Thickness model
- RSU model
- optional family display names if easy

C. For preset bundles, show current definitions clearly.
Example:
- `balanced_default`
  - RS = gbr
  - Thickness = gbr
  - RSU = gbr

D. Requirements:
1. Keep the current multi-select bundle list.
2. Do not add editing yet in this prompt.
3. The details panel should update on selection change.
4. If multiple bundles are selected, show:
   - first selected bundle details
   - or a compact list view if easier
5. Do not break existing benchmark flow.

E. Optional backend helper:
If useful, add:
- `get_bundle_catalog()`
- `describe_bundle(bundle_name)`

F. Add tests if practical for any pure helper logic.

Output:
- summarize where the bundle details panel appears
- explain how users can inspect bundle definitions