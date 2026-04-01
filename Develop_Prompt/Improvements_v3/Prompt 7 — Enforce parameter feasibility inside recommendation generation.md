Integrate the new parameter constraints into recommendation generation. Do not change unrelated behavior.

Goal:
Recommended recipes must be feasible according to:
- parameter type
- min/max
- step size
- dependent rules like incident_angle vs linear_offset

Requirements:
1. Identify the current recommendation generation path.
2. Before final recommendation output, validate every candidate parameter set.
3. Add a feasibility layer that can:
   - reject infeasible candidates
   - optionally repair values by clamping/snapping when safe
   - record why a candidate was rejected or changed
4. Surface this information in the recommendation UI:
   - final feasible recommendation
   - any corrected values
   - any skipped candidate reasons
5. Keep prediction separate from feasibility:
   - prediction can score candidates
   - feasibility filter decides whether candidate is allowed
6. Make sure dependent rules are respected during search/optimization, not only at the final display step if practical.

Acceptance criteria:
- The recommendation output always respects parameter constraints.
- Rejected or adjusted values have visible reasons.
- Dependent rules affect recommendation feasibility correctly.