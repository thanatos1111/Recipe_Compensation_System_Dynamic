Build a reusable benchmark runner so every future modeling method can be evaluated the same way.

Goals:
- Run benchmark suites across multiple candidate models
- Compare models fairly under the same split policy
- Produce structured benchmark tables for the UI

Implement the following:

A. Create:
- `core/benchmarking.py`

Add:
1. `run_prediction_benchmark(df, *, config, model_names_by_target, split_modes) -> BenchmarkSuiteResult`
2. `benchmark_single_model_bundle(df, *, config, rs_model_name, thickness_model_name, rsu_model_name, split_mode) -> BenchmarkSummary`
3. `rank_benchmark_results(results, primary_metric="spec_pass_accuracy", secondary_metric="rs_mae")`

B. Split modes to support immediately:
- `forward_chaining`
- `leave_one_target_out`
- `active_target_cutoff`

C. Benchmark outputs:
For each model bundle and split mode, compute:
- mean and std of RS MAE
- mean and std of Thickness MAE
- mean and std of RSU MAE
- mean and std of RMSE values
- mean spec-pass accuracy
- fold count
- average train row count
- average test row count
- warnings

D. Add simple model bundle presets:
- baseline_linear
- baseline_tree
- balanced_default
- mlp_experimental

Example preset definitions:
- baseline_linear = linear / linear / linear
- baseline_tree = rf / rf / rf
- balanced_default = gbr / gbr / gbr
- mlp_experimental = mlp / mlp / mlp

E. Add tests:
- `tests/test_benchmarking.py`
Test:
1. multiple model bundles run without crashing
2. ranking returns deterministic order
3. split-mode labels are preserved
4. fold aggregation is correct

Design rules:
- Use only leakage-free splits
- Do not change recommendation logic yet
- Keep benchmark outputs serializable and UI-friendly
- Keep future external models easy to plug into the same runner

Output:
- Add benchmark runner
- Add preset model bundles
- Add tests
- Summarize how benchmarks can now be reused by future model methods