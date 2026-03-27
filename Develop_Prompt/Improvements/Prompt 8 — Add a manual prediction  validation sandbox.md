Follow the existing project structure and the latest clarified design: each sheet is one material, rows are grouped by Target ID into target instances, different materials must not be mixed, and recommendations for the active target instance may use same-material historical target instances with target-instance identity preserved. Keep changes scoped strictly to this request. Do not refactor unrelated modules unless necessary to make this feature work.

Add a manual prediction sandbox that lets the user enter a lifetime and recipe combination and see predicted RS, Thickness, and RSU.

Goal:
This should be usable both as a what-if simulator and as a model validation tool when actual measured results are available later.

Implement the following:
1. Add a new panel or subsection called “Prediction Sandbox” or similar
2. Let the user enter:
   - lifetime
   - recipe parameter values
   - optionally select material and active target instance context
3. Run the trained model and show:
   - predicted RS
   - predicted Thickness
   - predicted RSU
   - in-spec probability if available
   - confidence / extrapolation warning
4. Add an optional actual-result entry section where the user can later input:
   - actual RS
   - actual Thickness
   - actual RSU
5. Then compute and show:
   - prediction error
   - pass/fail comparison
   - whether the model was directionally correct
6. Store these validation comparisons for later analysis
7. Add guidance text:
   - if prediction and actual mismatch badly, what likely causes exist
   - data sparsity
   - machine change
   - model drift
   - wrong feature assumptions
   - insufficient current-target correction

Implementation guidance:
- Keep prediction sandbox separate from optimizer
- Reuse prediction pipeline from existing trained model
- Add tests for manual prediction input and error comparison

Output:
- Add manual prediction and validation workflow
- Make it usable for future real-world checking