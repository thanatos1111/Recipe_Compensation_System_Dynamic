Implement a structured parameter-definition system for this app. Do not change unrelated behavior.

Goal:
Replace the current loose/script-text style settings for process parameters with a real data model that supports:
- parameter display name
- canonical internal name
- aliases/header names from Excel
- data type: continuous / discrete_step / categorical / boolean
- min value
- max value
- step size / minimum unit
- enabled flag
- notes / description
- optional unit
- optional dependent constraints

Requirements:
1. Add a new backend module for parameter definitions and constraints.
2. Create typed models/classes/dataclasses for:
   - ParameterDefinition
   - ParameterAlias
   - SimpleRangeConstraint
   - DependentRangeConstraint
   - ParameterRegistry
3. The registry must support lookup by:
   - canonical name
   - display name
   - any alias from Excel header row
4. Add persistence for this registry using a structured format already suitable for the app (prefer JSON or SQLite table(s), whichever fits current architecture best). Do not use raw editable script text.
5. Keep backward compatibility if older settings exist. If needed, add a migration loader that converts old settings into the new registry structure.
6. Add validation methods:
   - validate a single parameter definition
   - validate an input parameter value against min/max/step
   - return a structured validation result, not only a boolean
7. Keep this prompt backend-only unless a tiny bridge is needed for existing UI. No full settings UI redesign yet.

Acceptance criteria:
- I can define parameters in a structured registry.
- A parameter can be found by Excel title alias.
- Step/min/max validation works from the new backend.
- The code is clean and ready for later Settings UI integration.
- Existing app behavior outside settings/parameter-definition logic is unchanged.