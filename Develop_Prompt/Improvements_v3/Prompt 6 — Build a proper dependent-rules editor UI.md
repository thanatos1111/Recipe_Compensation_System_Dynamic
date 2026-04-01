Now add full UI editing for dependent parameter rules in Settings. Do not change unrelated behavior.

Goal:
Users should be able to define special rules like incident_angle ranges depending on linear_offset without editing raw text.

Requirements:
1. In the Settings dialog, implement the Dependent Rules tab fully.
2. Add a rule table/grid with columns like:
   - enabled
   - target parameter
   - driver parameter
   - driver min
   - driver max
   - target min
   - target max
   - priority
   - notes
3. Add Add / Edit / Delete / Duplicate controls.
4. Add input validation in the editor:
   - target and driver must exist in parameter registry
   - numeric ranges must be valid
   - overlapping rules should show warning if ambiguous
5. Add a preview/example section:
   - user picks a driver value
   - UI shows which target range becomes active
6. Persist these rules using the backend structure from previous prompts.

Acceptance criteria:
- I can define incident_angle vs linear_offset rules visually.
- The rules are saved and reloaded.
- Ambiguous or invalid rules are flagged in UI.