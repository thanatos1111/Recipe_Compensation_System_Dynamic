Implement the next milestone for `recipe_comp_system`: append new run data for the selected active target instance and refresh all related artifacts.

Scope:
- Add new row(s) to one target instance only.
- Recompute derived fields and labels.
- Rebuild baseline trend artifacts.
- Retrain or refresh material-level models if needed.
- Refresh target-instance correction.
- Refresh recommendation state.

Implement:
1. `core/update_pipeline.py` with append and refresh functions.
2. `ui/update_panel.py` for manual row input or CSV append.
3. Target-instance-specific retrain trigger.
4. Optional artifact snapshot saving.
5. Tests for append + refresh behavior.

Keep it simple and deterministic. Do not introduce cross-material learning.