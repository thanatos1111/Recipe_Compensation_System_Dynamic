Implement dependent parameter constraints for cases like incident_angle being limited by the value range of linear_offset. Do not change unrelated behavior.

Goal:
Support rules like:
- if linear_offset is in range A, then incident_angle must be in range X
- if linear_offset is in range B, then incident_angle must be in range Y

Requirements:
1. Add a general dependent-constraint model, not a one-off hardcoded incident_angle hack.
2. Support rules with:
   - target parameter (example: incident_angle)
   - driver parameter (example: linear_offset)
   - driver range min/max
   - allowed target range min/max
   - optional priority/order
   - optional inclusive/exclusive range flags if needed
3. Add validation logic:
   - given a parameter set, validate target parameter against all applicable dependent rules
   - if multiple rules match, handle deterministically
   - if no rule matches, fall back to base min/max if defined
4. Add a readable explanation string for violations, e.g.:
   "incident_angle=35 is invalid because when linear_offset is between 10 and 20, incident_angle must be between 15 and 25"
5. Keep it generic so future dependent rules can be added for other parameters too.

Acceptance criteria:
- The app can enforce incident_angle ranges that depend on linear_offset ranges.
- The mechanism is generic and reusable.
- Violations produce clear explanations.