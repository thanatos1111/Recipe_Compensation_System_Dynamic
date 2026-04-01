"""Tests for scoped material/target settings and default presets."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.config_store import save_json
from core.scoped_settings import (
    apply_scoped_parameter_constraints_to_config,
    default_scoped_settings_dict,
    is_shipped_target_position_preset,
    load_effective_scoped_settings,
    matching_spec_profiles_ordered,
    matching_parameter_profiles_ordered,
    merge_parameter_constraints_delta,
    resolve_effective_spec_dict,
    resolve_parameter_override_delta,
    target_position_from_target_id,
    validate_scoped_parameter_profile,
)


class TestScopedSettings(unittest.TestCase):
    def test_target_position_from_material_dot_number(self) -> None:
        self.assertEqual(target_position_from_target_id("Ta", "Ta.7"), 7)
        self.assertEqual(target_position_from_target_id("Foo", "Foo.12"), 12)

    def test_target_position_plain_digit(self) -> None:
        self.assertEqual(target_position_from_target_id("x", "3"), 3)

    def test_default_presets_count(self) -> None:
        d = default_scoped_settings_dict()
        self.assertEqual(len(d["parameter_profiles"]), 12)

    def test_shipped_preset_validation_skipped(self) -> None:
        p = {"id": "preset-tp-1", "material_name": "", "target_position": 1, "parameter_overrides": {}}
        self.assertTrue(is_shipped_target_position_preset(p))
        self.assertEqual(validate_scoped_parameter_profile(p), [])

    def test_matching_profiles_wildcard_material(self) -> None:
        scoped = default_scoped_settings_dict()
        profs = matching_parameter_profiles_ordered(scoped, "AnySheet", 1)
        self.assertTrue(any(p.get("id") == "preset-tp-1" for p in profs))

    def test_matching_spec_profiles_material_glob(self) -> None:
        scoped = {
            "version": 1,
            "parameter_profiles": [],
            "spec_profiles": [
                {"id": "s1", "enabled": True, "material_name": "ta*", "target_position": 7, "priority": 0, "spec": {"rs_target": 1.0}},
            ],
        }
        m = matching_spec_profiles_ordered(scoped, "Ta (sheet)", 7)
        self.assertEqual([p.get("id") for p in m], ["s1"])

    def test_resolve_effective_spec_precedence_material_then_target(self) -> None:
        cfg = {
            "spec_settings": {"rs_target": 10.0, "use_rs_spec": True},
            "material_overrides": {"materials": {"Ta": {"spec_settings": {"rs_target": 999.0}}}},
        }
        scoped = {
            "version": 1,
            "parameter_profiles": [],
            "spec_profiles": [
                {"id": "mat", "enabled": True, "material_name": "Ta", "priority": 0, "spec": {"rs_target": 20.0}},
                {"id": "tp", "enabled": True, "material_name": "Ta", "target_position": 7, "priority": 0, "spec": {"rs_target": 30.0}},
            ],
        }
        # Material-only profile should override legacy material override.
        out_mat = resolve_effective_spec_dict(cfg, scoped, material_name="Ta", target_id=None)
        self.assertEqual(out_mat.get("rs_target"), 20.0)
        # Target-position profile should override material-only.
        out_tp = resolve_effective_spec_dict(cfg, scoped, material_name="Ta", target_id="Ta.7")
        self.assertEqual(out_tp.get("rs_target"), 30.0)

    def test_scoped_delta_merges_linear_offset(self) -> None:
        scoped = default_scoped_settings_dict()
        profs = matching_parameter_profiles_ordered(scoped, "M", 5)
        delta = resolve_parameter_override_delta(profs)
        self.assertIn("linear_offset", delta)
        self.assertEqual(delta["linear_offset"]["min_value"], -15.0)
        self.assertEqual(delta["linear_offset"]["max_value"], 190.0)

    def test_apply_scoped_updates_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir(parents=True)
            save_json(root / "config" / "default_parameter_registry.json", {})
            save_json(root / "config" / "user_parameter_registry.json", {})
            save_json(root / "config" / "default_config.json", {"parameter_constraints": {}})
            save_json(root / "config" / "user_config.json", {})

            from core.parameter_registry import apply_parameter_registry_to_config

            cfg: dict = {"parameter_constraints": {}, "column_mapping": {}}
            apply_parameter_registry_to_config(root, cfg)
            scoped = default_scoped_settings_dict()
            apply_scoped_parameter_constraints_to_config(
                root,
                cfg,
                material_name="Mat",
                target_id="Mat.1",
                scoped=scoped,
            )
            lo = (cfg.get("parameter_constraints") or {}).get("linear_offset") or {}
            self.assertEqual(lo.get("min_value"), -190.0)
            self.assertEqual(lo.get("max_value"), 15.0)

    def test_empty_user_parameter_profiles_keeps_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir(parents=True)
            save_json(root / "config" / "default_scoped_settings.json", default_scoped_settings_dict())
            merged = load_effective_scoped_settings(
                root,
                {"scoped_settings": {"parameter_profiles": [], "version": 1}},
            )
            self.assertEqual(len(merged.get("parameter_profiles") or []), 12)

    def test_merge_parameter_constraints_replaces_dependent(self) -> None:
        base = {
            "incident_angle": {
                "type": "continuous",
                "dependent_constraints": [{"driver_parameter": "x", "rules": []}],
            }
        }
        delta = {
            "incident_angle": {
                "dependent_constraints": [{"driver_parameter": "linear_offset", "rules": [{"driver_min": 0, "driver_max": 1}]}]
            }
        }
        out = merge_parameter_constraints_delta(base, delta)
        self.assertEqual(out["incident_angle"]["dependent_constraints"][0]["driver_parameter"], "linear_offset")


if __name__ == "__main__":
    unittest.main()
