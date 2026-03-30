Extend the Model tab so it can run and display benchmark comparisons across multiple model methods using the new benchmark runner.

Goals:
- Keep current single-model training flow
- Add a separate benchmark workflow
- Show fold-safe benchmark results clearly
- Make it obvious which model is best under which split mode

Implement the following in `ui/model_panel.py` and supporting helpers:

A. Add a new sub-tab:
- `5) Benchmark`

B. Add controls:
1. Split mode selector:
- forward_chaining
- leave_one_target_out
- active_target_cutoff
2. Model bundle multi-select or checkbox list:
- baseline_linear
- baseline_tree
- balanced_default
- mlp_experimental
3. Run benchmark button
4. Option to choose active target and cutoff lifetime for active_target_cutoff mode

C. Add result views:
1. Summary table:
- model bundle
- split mode
- RS MAE
- Thickness MAE
- RSU MAE
- spec-pass accuracy
- fold count
- warnings
2. Detailed fold table
3. Comparison plot:
- grouped bar plot or scatter plot for selected metrics
4. Best-model explanation panel:
- why this model ranked highest
- what tradeoffs exist
- whether confidence is low due to data sparsity

D. Add benchmark export support:
- CSV export of summary table
- CSV export of fold table

E. Add UI logic so benchmark output does not overwrite the deployment-trained model unless the user explicitly chooses to adopt a benchmark winner later

F. Add tests where practical for helper formatting functions

Design rules:
- Keep the existing training panel functioning
- Benchmark must use the leakage-free evaluation pipeline
- Make the displayed labels concise and readable
- Show clear warnings when row counts are too small

Output:
- Add Benchmark sub-tab to Model panel
- Wire it to the new benchmark runner
- Add summary + fold details + chart
- Keep deployment training separate from benchmarking