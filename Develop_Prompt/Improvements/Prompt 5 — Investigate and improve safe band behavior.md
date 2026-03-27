Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Investigate and improve the safe band logic in the recommendation section.

Observed issue:
- Only one safe-band parameter appears in the recommendation section
- The safe-band min/max range may be larger than the range represented in the recommended candidate list

I want both diagnosis and improvement.

Implement the following:
1. Audit the current safe-band logic and explain:
   - why only one parameter is currently shown
   - how the safe band is computed
   - why its range may exceed the values seen in the candidate list
2. Add a debug or explanation section in the recommendation tab showing:
   - safe-band calculation method
   - search space used
   - whether safe band is based on one-parameter variation while keeping others fixed
   - whether candidate list uses a narrower search neighborhood
3. Improve safe-band behavior so that:
   - it can show safe bands for multiple key parameters, not only one
   - it clearly distinguishes between:
     a. feasible parameter range
     b. currently enumerated candidate range
4. Add UI labels to avoid confusion:
   - “feasible band”
   - “candidate search range”
   - “recommended center value”
5. If needed, add config options for:
   - which parameters get safe-band analysis
   - search width for safe-band estimation
   - whether safe-band range is clipped to optimizer neighborhood

Implementation guidance:
- Do not silently change math without showing the reason
- Keep one-parameter-at-a-time safe-band logic if needed initially, but make it explicit
- Add tests for safe-band range generation and consistency checks

Output:
- Diagnose current safe-band behavior
- Improve the UI and logic so users can understand it clearly