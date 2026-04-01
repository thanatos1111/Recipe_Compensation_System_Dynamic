Prompt 2.1 — Add scoped material+target settings for parameter constraints and D.Spec profiles

Goal:
Extend the new tabbed Settings system so the app supports scoped settings by:
- material name
- target position

This is needed because parameter limits are not only global. In actual use, constraints for `linear_offset` and `incident_angle` depend on target position, and later may also vary by material. Also, D.Spec config should be managed in the Settings UI and loaded automatically when matching material/target data is selected.

Important:
- Do not replace the global parameter registry. Keep it as the base/default layer.
- Add a clean scoped override layer on top of it.
- Do not reintroduce raw JSON/script editing.
- Keep existing behavior working for users who only use global settings.

Scope model:
Implement a settings precedence model like this:

For parameter constraints:
1. global parameter registry defaults
2. material-level scoped overrides
3. material + target-position scoped overrides

For D.Spec config:
1. global spec defaults
2. material-level spec profile
3. material + target-position spec profile

Target position rules:
- target positions are integers 1 to 12
- the user must be able to assign/edit scoped profiles using material name + target position
- use current target identifier/instance if needed, but normalize the concept in the settings layer as `target_position`
- if raw data already identifies target position, use that
- if the app currently stores target as another field, add a bridge so the settings layer still works with target positions 1–12 cleanly

What to add:

1) New backend support for scoped settings profiles
Add a structured backend model for:
- material-scoped parameter constraint profile
- material+target-position parameter constraint profile
- material-scoped spec profile
- material+target-position spec profile

Each scoped parameter profile must support:
- material_name
- optional target_position
- enabled flag
- notes
- per-parameter overrides for min/max/step
- dependent rules for incident_angle based on linear_offset span
- future-safe structure so more scoped parameters can be added later

Each scoped spec profile must support:
- material_name
- optional target_position
- full D.Spec config fields
- enabled flag
- notes

Persist these in a structured way that fits current architecture cleanly.
Do not create a fragile one-off ad hoc config path.

2) Settings UI additions
Extend the Settings dialog with proper editable tabs/panels for:
- Scoped Parameter Profiles
- Scoped Spec Profiles

Scoped Parameter Profiles UI:
- table/list of profiles
- columns/fields: enabled, material_name, target_position, notes
- add/edit/delete/duplicate actions
- selecting a profile shows editable parameter overrides
- user can edit at least:
  - linear_offset min/max/step
  - incident_angle dependent rules
- show whether the profile is material-only or material+target-position scoped

Scoped Spec Profiles UI:
- table/list of spec profiles
- columns/fields: enabled, material_name, target_position, notes
- add/edit/delete/duplicate actions
- editing shows full D.Spec fields, not just partial placeholders

3) Default target-position constraint presets
Seed the system with default scoped target-position rules.

Linear offset defaults:
- target 1, 2, 7, 8: linear_offset range = -190 to +15
- target 3, 4, 9, 10: linear_offset range = -190 to +15
- target 5, 6, 11, 12: linear_offset range = -15 to +190

Incident angle defaults:

A. target 1, 3, 7, 9
- incident_angle range = 0 to 46 when linear_offset is in [-190, -175]
- incident_angle range = 0 to 50 when linear_offset is in [-175, +15]

B. target 5, 11
- incident_angle range = 0 to 46 when linear_offset is in [+175, +190]
- incident_angle range = 0 to 50 when linear_offset is in [-15, +175]

C. target 2, 4, 8, 10
- incident_angle range = 18 to 78 when linear_offset is in [-190, -175]
- incident_angle range = 14 to 78 when linear_offset is in [-175, +15]

D. target 6, 12
- incident_angle range = 18 to 78 when linear_offset is in [+175, +190]
- incident_angle range = 14 to 78 when linear_offset is in [-15, +175]

Implementation notes:
- these should be represented as scoped dependent rules, not hardcoded if/else hacks in runtime logic
- use inclusive numeric spans unless current architecture requires otherwise
- if a target-position scoped rule exists, it should override material-only and global defaults
- if no scoped rule exists, fall back to global parameter registry behavior

4) Effective constraint resolution
Add a resolver/helper that computes the effective runtime constraints from:
- global registry
- selected material
- selected target position

This resolver should output effective constraints used by:
- recommendation generation
- candidate feasibility checks
- parameter validation in UI
- any training/preprocessing logic that needs parameter bounds

The resolver must especially handle:
- effective `linear_offset` min/max/step
- effective `incident_angle` dependent ranges based on current `linear_offset`
- material+target-position override precedence

5) Runtime integration
When a workbook/raw dataset is loaded and the user selects a material + target:
- auto-resolve the effective scoped parameter constraints
- auto-resolve the effective scoped D.Spec config
- apply them in the current panels
- D.Spec panel should display the matching scoped spec if one exists
- recommendation/runtime validation should use the matching scoped parameter constraints

6) D.Spec profile management in Settings
The current D.Spec config should be manageable from the Settings UI too.

Add support to:
- list spec profiles by material and optional target position
- add new spec profiles
- edit spec profiles
- remove spec profiles
- load matching spec profile automatically when raw data selection changes
- preserve existing material override behavior if already present, but migrate/bridge it into the new structured settings model where practical

Include full D.Spec fields already used in the app, such as:
- RS target / min / max / tolerance
- thickness target / min / max / tolerance
- RSU max
- flags like use_rs_target_mode, use_thickness_target_mode, use_rs_spec, use_thickness_spec, use_rsu_spec
- any max lifetime field currently coupled with spec override logic if appropriate

7) Validation and UX requirements
- validate target_position is 1–12
- validate material_name is non-empty
- validate numeric ranges are correct
- warn on overlapping ambiguous dependent rules within the same scope
- show field-level validation errors in UI
- do not crash on incomplete rows
- keep the UI consistent with the new tabbed Settings style

8) Backward compatibility
- keep existing global-only settings working
- if older spec overrides exist in user config, bridge them into the new scoped model or read them compatibly
- do not break users who never define material/target-specific settings

9) Cleanups
While implementing this, also clean up the Prompt 2 issue where a user-local mutable settings file was committed to the repo if that is still present. User-local runtime registry/profile files should not become inappropriate tracked defaults unless intentionally designed as seed defaults.

Acceptance criteria:
- I can open Settings and create/edit/delete scoped parameter profiles by material and target position
- I can define target-position-specific linear_offset ranges
- I can define target-position-specific incident_angle ranges that depend on linear_offset span
- the provided default target-position presets are available by default
- I can open Settings and create/edit/delete scoped D.Spec profiles by material and optional target position
- when I load/select a material and target in the app, the matching scoped parameter constraints and D.Spec config are auto-loaded
- recommendation and validation use the effective scoped constraints
- global-only users still work without needing scoped profiles
- implementation is structured and extensible, not a one-off hardcoded patch