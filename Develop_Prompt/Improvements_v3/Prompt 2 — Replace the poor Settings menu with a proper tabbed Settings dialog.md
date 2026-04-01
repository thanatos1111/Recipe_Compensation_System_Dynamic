Redesign the Settings function into a proper tabbed/panel-based UI instead of raw script-text editing. Do not change unrelated parts of the app.

Goal:
Create a clean Settings dialog/window with tabs so users can edit model/process parameter settings through normal controls.

Required tabs:
1. General
2. Parameters
3. Parameter Aliases / Excel Matching
4. Dependent Rules
5. Model / Validation Defaults

Requirements:
1. Replace or wrap the current Settings entry point with a modern tabbed UI.
2. In the Parameters tab, show a table/grid with editable columns:
   - enabled
   - display name
   - canonical name
   - unit
   - type
   - min
   - max
   - step
   - notes
3. Allow add / edit / delete / reorder parameters through buttons and dialogs or inline editing.
4. In the Aliases tab, allow mapping multiple Excel header names to one canonical parameter.
5. In the Dependent Rules tab, leave placeholders or read-only support for now if full editing is not ready yet; the UI structure must exist.
6. In the Model / Validation Defaults tab, add placeholders/settings area for future validation options like:
   - default split mode
   - default cutoff mode
   - default metrics to show
7. Add Save, Cancel, Reset, and Apply buttons with safe validation before saving.
8. Show field-level validation errors in the UI instead of crashing.
9. Keep the visual style consistent with the rest of the app, but make it clearly better than raw script text editing.

Acceptance criteria:
- Settings opens as a normal tabbed dialog/window.
- Parameter definitions can be edited through tables/forms.
- Alias editing is possible in UI.
- Save/load works using the new backend registry from Prompt 1.
- No unrelated functions are broken.