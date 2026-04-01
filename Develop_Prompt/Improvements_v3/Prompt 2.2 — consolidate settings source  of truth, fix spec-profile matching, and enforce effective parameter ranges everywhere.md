Prompt 2.2 — consolidate settings source of truth, fix spec-profile matching, and enforce effective parameter ranges everywhere

Goal:
Stabilize the settings system before later prompts. Right now range/step settings are duplicated across General, Parameters, and Scoped Parameter Profiles, spec-profile matching is not fully working, and effective parameter ranges are not guaranteed to be enforced everywhere in the app.

Requirements:

1) Remove duplicated range/step settings
- Make one clear source of truth for base parameter bounds:
  - Parameters tab = canonical parameter definition + base min/max/step
- Scoped Parameter Profiles tab = only material / target-position overrides on top of base values
- Alias tab = Excel header matching only
- General tab should no longer be a second editable source for parameter step/range
- If current General minimum-steps table is still needed for backward compatibility, convert it to migration/read-only support or remove it cleanly
- Do not keep multiple editable places for the same parameter range/step logic

2) Fix spec-profile matching and effective selection
- Fix the issue where only the first spec profile can overwrite defaults but later profiles do not take effect
- Ensure matching works correctly for:
  - material-only spec profiles
  - material + target-position spec profiles
- Verify deterministic precedence:
  - global spec defaults
  - material-level spec profile
  - material + target-position spec profile
- Ensure the correct matching profile is applied for the currently selected material and target
- Check the actual target-id / target-position bridge used by imported workbook data and make matching reliable for real app data, not just ideal test strings

3) Make effective parameter ranges apply everywhere
- Once effective constraints are resolved for the active material + target position, enforce them everywhere relevant in the app
- No impossible recipe parameter outside the effective allowed range should appear in:
  - recommendation outputs
  - candidate generation / optimization search
  - manual parameter editing/input UI
  - runtime validation
  - any other place where recipe parameter values are created or changed
- This must especially work for:
  - linear_offset base min/max/step
  - incident_angle dependent ranges based on linear_offset span
- If a value is invalid, either clamp/snap safely or reject it with a clear reason, but do not silently let impossible values propagate

4) Keep settings model simple
- Final intended settings structure should be:
  - Parameters tab = base min/max/step/type
  - Aliases tab = Excel header aliases
  - Scoped Parameter Profiles = material/target-position-specific overrides, including incident_angle vs linear_offset bands
  - Scoped Spec Profiles = D.Spec per material / target-position
- Do not add more overlapping config paths

5) Validation / cleanup
- Add or update tests for:
  - spec profile precedence and matching
  - target-position-specific scoped parameter resolution
  - effective range enforcement in recommendation/runtime paths
- Clean up any stale settings behavior left from earlier prompts
- Keep existing global-only behavior working

Acceptance:
- There is only one editable base place for parameter min/max/step
- Scoped profiles only act as overrides
- Spec profiles match correctly for the selected material/target, not just the first profile
- No out-of-range recipe parameters appear anywhere in the app after effective scoped constraints are resolved
- linear_offset and incident_angle constraints behave correctly for target-position-dependent rules
- implementation stays clean and extensible