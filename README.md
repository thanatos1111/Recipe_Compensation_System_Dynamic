# Recipe Parameter Compensation System

## Purpose
Material-specific, target-instance-aware process recipe compensation for plasma deposition.

This project ingests historical deposition records from an Excel workbook, labels runs against specification, visualizes trends, and (in later milestones) trains predictive models and proposes discrete recipe recommendations.

## Milestone 1 Scope (Prompt 1)
This milestone establishes a robust project scaffold only:
- Top-level project structure (`app.py`, `core/`, `ui/`, `config/`, `artifacts/`, `tests/`)
- Placeholder modules in `core/` and `ui/`
- Typed schema dataclasses in `core/schemas.py`
- Minimal PySide6 app shell that starts a window (panels are placeholders)
- Basic `unittest` tests to verify imports and schema construction

## Folder Structure
```text
recipe_comp_system/
  app.py
  requirements.txt
  README.md
  core/
    importer.py
    schemas.py
    labeling.py
    validation.py
    trend_fitting.py
    feature_engineering.py
    response_models.py
    instance_correction.py
    candidate_generator.py
    constraints.py
    optimizer.py
    safe_band.py
    update_pipeline.py
    exports.py
  ui/
    main_window.py
    material_selector.py
    target_instance_selector.py
    raw_table_panel.py
    spec_config_panel.py
    trend_panel.py
    model_panel.py
    recommendation_panel.py
    update_panel.py
  config/
    default_config.json
  artifacts/
  tests/
```

## How to Run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the app:
   - `python app.py`

## Notes / Assumptions
- This milestone intentionally does not implement business logic. Functions/classes currently raise `NotImplementedError` where appropriate.
- `core/` modules are UI-framework-agnostic to keep later milestone logic testable.

## Milestone 2 Scope (Prompt 2)
- Implements Excel workbook loading and per-sheet material isolation.
- Normalizes raw columns into canonical fields using `config/default_config.json` column mapping.
- Groups rows into `TargetInstance` objects via the `Target ID` column.
- Adds importer + grouping/unit tests.

