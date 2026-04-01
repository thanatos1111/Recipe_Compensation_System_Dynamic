Implement Excel header-to-parameter matching using the new parameter registry. Do not change unrelated behavior.

Goal:
When loading input data from Excel, the app should recognize parameter columns by matching the Excel title row against the parameter settings.

Requirements:
1. Identify the current code path where Excel title/header rows are read.
2. Add a normalization + matching layer that maps raw Excel headers to canonical parameter names using:
   - exact canonical name
   - exact display name
   - alias match
   - normalized match (case-insensitive, trimmed, underscore/space tolerant)
3. Expose a mapping result object that includes:
   - raw header
   - matched canonical parameter
   - confidence / match type
   - unmatched flag
4. Add a lightweight UI/report summary when data is loaded:
   - matched parameters
   - unmatched headers
   - duplicate matches
5. Make sure existing downstream code uses canonical parameter names instead of raw Excel strings wherever practical.
6. Do not silently discard unmatched headers. Keep them available for inspection.

Acceptance criteria:
- The app can match Excel parameter titles to settings-defined parameters.
- Similar names can be matched through aliases/normalization.
- Unmatched headers are visible to the user.
- Downstream logic can rely on canonical parameter names.