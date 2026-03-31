Add custom benchmark bundle creation and persistence in `Recipe_Compensation_System_Dynamic`.

Goal:
Allow users to define their own RS / Thickness / RSU model bundles and use them in manual benchmark, manual adoption, and later auto-decision.

Implement the following:

A. Add backend support
Files:
- create `core/bundles.py` if not already present
- update config handling as needed

B. Support bundle catalogs with two sources:
1. preset bundles
2. custom bundles

C. Add config persistence:
- store custom bundles in config, e.g.:
  - `custom_model_bundles`
Each custom bundle should define:
- `rs`
- `thickness`
- `rsu`

D. Add UI in Benchmark section:
1. `Create custom bundle`
2. `Edit custom bundle`
3. `Delete custom bundle` for user-defined bundles only

E. Add a simple bundle editor dialog:
Fields:
- bundle name
- RS model dropdown
- Thickness model dropdown
- RSU model dropdown

Use all currently supported registry model names.

F. Requirements:
1. Merge preset + custom bundles into the benchmark list.
2. Prevent name collisions with preset bundles unless explicitly handled with a clear warning.
3. Validate selected model names.
4. Keep preset bundles read-only.
5. Keep custom bundles available for benchmark and adopt-winner flow.

G. Update bundle details panel so it shows whether bundle source is preset/custom.

H. Add tests:
- `tests/test_bundles.py`

Test:
1. custom bundle validation
2. preset + custom bundle merge
3. duplicate name handling
4. config serialization/deserialization if applicable

Output:
- summarize custom bundle workflow
- explain how custom bundles are stored and reused