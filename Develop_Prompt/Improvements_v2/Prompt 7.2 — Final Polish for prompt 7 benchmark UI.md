Do a short final polish pass for the external boosted-model benchmark UI and warning handling in `Recipe_Compensation_System_Dynamic`.

Goals:
1. Improve how benchmark warnings are shown in the Benchmark tab’s best-explanation / status area.
2. Clean up small leftover issues from Prompt 7.1.
3. Keep behavior unchanged unless it improves clarity.

Implement the following:

A. Improve benchmark warning display in the UI
Files:
- `ui/model_panel.py`

Requirements:
1. In the Benchmark section, add a clearly visible warning summary block near the best-model explanation / status area.
2. If benchmark results contain aggregate warnings (for example unavailable external model dependencies), display them in a readable multi-line format.
3. If multiple bundles have the same repeated warning, deduplicate the displayed warning lines.
4. Keep the existing tables and plots unchanged.
5. Do not hide successful benchmark results just because one bundle has warnings.

Suggested behavior:
- Add a small read-only text area or QLabel section titled `Benchmark warnings`
- Show `None` or hide the block when there are no warnings
- Prefer concise formatting like:
  - `xgb_default: missing dependency xgboost`
  - `lgbm_default: warning ...`

B. Improve backtest warning display in the UI
Files:
- `ui/model_panel.py`

Requirements:
1. The backtest table already includes a `warnings` column; also surface a compact warning summary below or above the table.
2. Deduplicate repeated warnings across bundles.
3. If a bundle has warnings and no useful backtest rows, make that easy to understand in the summary text.

C. Clean up small leftover code issues
Files:
- `ui/model_panel.py`
- any other small helper locations if needed

Requirements:
1. Remove pointless no-op code like:
   - `missing_error.replace("pip install ", "pip install ")`
2. Replace it with direct, clean formatting logic.
3. Keep existing functionality the same.
4. Avoid adding unnecessary abstraction.

D. Add small helper functions if useful
Examples:
- `_format_warning_lines(warnings: list[str]) -> list[str]`
- `_dedupe_preserve_order(items: list[str]) -> list[str]`

Keep helpers local and lightweight.

E. Add/update tests if practical
Files:
- add a small UI/helper test file only if easy
- otherwise test pure formatting/helper functions

Test ideas:
1. duplicate warnings are deduplicated but original order is preserved
2. empty warnings render as empty/hidden state
3. warning formatting is readable and stable

Constraints:
- Do not redesign benchmark ranking
- Do not change model behavior
- Do not change backtest math
- This is a polish-only pass

Output:
- summarize files changed
- describe how benchmark warnings now appear in the UI
- mention any leftover limitations