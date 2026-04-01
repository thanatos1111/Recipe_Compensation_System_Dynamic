Extend the parameter handling so each numeric parameter supports min/max limits in addition to the existing minimum-unit / stepwise behavior. Do not change unrelated behavior.

Goal:
Parameters currently support stepwise minimum unit. Add full limitation support:
- minimum allowed value
- maximum allowed value
- step size enforcement

Requirements:
1. Apply min/max/step validation consistently in:
   - settings editing
   - recommendation candidate generation
   - any manual parameter input UI
   - training/recommendation preprocessing where relevant
2. For continuous parameters:
   - allow min/max
   - step optional if current app logic supports true continuous values
3. For discrete_step parameters:
   - enforce min/max and step snapping
4. For categorical/boolean parameters:
   - use allowed-value validation, not numeric min/max
5. Add helper methods:
   - clamp_to_range
   - snap_to_step
   - validate_and_explain
6. In the recommendation flow, if a candidate violates constraints, either:
   - correct it safely, or
   - reject it with a clear reason
7. Add unit tests or equivalent coverage for edge cases:
   - below min
   - above max
   - off-step
   - invalid type

Acceptance criteria:
- Numeric parameters now support min/max limits throughout the app.
- Recommendation logic respects limits.
- The user can see clear validation/rejection reasons.