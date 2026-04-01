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