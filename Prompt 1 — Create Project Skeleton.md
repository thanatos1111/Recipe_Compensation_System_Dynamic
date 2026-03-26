Build the initial project skeleton for a Python desktop application named `recipe_comp_system` based on the technical design spec already discussed.

Requirements:
1. Create the following top-level structure:
   - app.py
   - requirements.txt
   - README.md
   - core/
   - ui/
   - config/
   - artifacts/
   - tests/
2. Inside `core/`, create placeholder modules with docstrings and minimal class/function stubs:
   - importer.py
   - schemas.py
   - labeling.py
   - validation.py
   - trend_fitting.py
   - feature_engineering.py
   - response_models.py
   - instance_correction.py
   - candidate_generator.py
   - constraints.py
   - optimizer.py
   - safe_band.py
   - update_pipeline.py
   - exports.py
3. Inside `ui/`, create placeholder modules:
   - main_window.py
   - material_selector.py
   - target_instance_selector.py
   - raw_table_panel.py
   - spec_config_panel.py
   - trend_panel.py
   - model_panel.py
   - recommendation_panel.py
   - update_panel.py
4. Create a `config/default_config.json` with placeholder keys for:
   - column mapping
   - spec settings
   - parameter constraints
   - fit settings
   - model settings
   - optimizer weights
5. In `core/schemas.py`, define typed dataclasses or pydantic-style classes for:
   - SpecConfig
   - ParameterConfig
   - MaterialDataset
   - TargetInstance
   - MaterialModelArtifacts
   - InstanceCorrectionArtifacts
   - FitArtifacts
   - RecommendationArtifacts
6. Implement a minimal `app.py` entry point that starts the application shell, even if panels are placeholders.
7. Populate `README.md` with:
   - project purpose
   - current milestone scope
   - folder structure
   - how to run
8. Add basic tests that validate imports and schema object construction.

Technical constraints:
- Use Python.
- Use pandas/numpy/scikit-learn/openpyxl/matplotlib in requirements.
- If a desktop UI framework is needed, use PySide6.
- Keep the code clean and implementation-ready, not pseudo-code.
- Do not implement full logic yet; only create a robust scaffold.

Output:
- Create all files.
- Show a concise summary of what was added.
- Note any assumptions explicitly in comments or README.