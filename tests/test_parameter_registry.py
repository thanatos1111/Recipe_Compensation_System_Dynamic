"""Tests for structured parameter registry, migration, and validation."""

from __future__ import annotations

import unittest
from pathlib import Path

from core.parameter_registry import (
    DependentRangeConstraint,
    DependentRangeRule,
    ParameterAlias,
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
    apply_parameter_registry_to_config,
    migrate_legacy_parameter_constraints,
    parameter_definition_to_legacy_entry,
    validate_parameter_definition,
    validate_parameter_value,
    validate_registry,
)


class TestParameterRegistry(unittest.TestCase):
    def test_lookup_by_alias_and_display(self) -> None:
        reg = ParameterRegistry()
        d = ParameterDefinition(
            canonical_name="incident_angle",
            display_name="Incident Angle",
            data_type=ParameterDataType.CONTINUOUS,
            aliases=[ParameterAlias(header="Incident Angle")],
        )
        reg.add(d)
        self.assertIs(reg.get_by_canonical("incident_angle"), d)
        self.assertIs(reg.get_by_alias_header("Incident Angle"), d)
        self.assertIs(reg.get_by_display_name("incident Angle"), d)
        self.assertIs(reg.resolve_header("Incident Angle"), d)

    def test_migration_matches_legacy_projection(self) -> None:
        legacy = {
            "rotations": {
                "type": "integer",
                "min_value": 0,
                "max_value": 10,
                "step": 1,
                "allowed_values": None,
                "is_enabled": True,
                "is_coupled": False,
                "coupling_group": None,
            }
        }
        column_mapping = {"Rotations": "rotations"}
        reg = migrate_legacy_parameter_constraints(legacy, column_mapping)
        out = reg.to_legacy_parameter_constraints()["rotations"]
        self.assertEqual(out["type"], "integer")
        self.assertEqual(out["step"], 1.0)
        self.assertEqual(out["min_value"], 0.0)
        self.assertEqual(out["max_value"], 10.0)

    def test_validate_value_step_and_min_max(self) -> None:
        d = ParameterDefinition(
            canonical_name="x",
            display_name="X",
            data_type=ParameterDataType.CONTINUOUS,
            min_value=0.0,
            max_value=10.0,
            step=0.5,
        )
        ok = validate_parameter_value(d, 2.0)
        self.assertTrue(ok.valid)
        bad = validate_parameter_value(d, 2.03)
        self.assertFalse(bad.valid)
        self.assertTrue(any(i.code == "not_on_step" for i in bad.issues))
        low = validate_parameter_value(d, -1.0)
        self.assertFalse(low.valid)

    def test_validate_definition_errors(self) -> None:
        bad = ParameterDefinition(
            canonical_name="",
            display_name="Y",
            data_type=ParameterDataType.CONTINUOUS,
            min_value=5.0,
            max_value=1.0,
        )
        r = validate_parameter_definition(bad)
        self.assertFalse(r.valid)

    def test_dependent_range_uses_driver(self) -> None:
        d = ParameterDefinition(
            canonical_name="child",
            display_name="Child",
            data_type=ParameterDataType.CONTINUOUS,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            dependent_constraints=[
                DependentRangeConstraint(
                    driver_parameter="driver",
                    rules=[
                        DependentRangeRule(driver_min=0.0, driver_max=1.0, min_value=0.0, max_value=5.0),
                    ],
                )
            ],
        )
        r = validate_parameter_value(d, 10.0, driver_values={"driver": 0.5})
        self.assertFalse(r.valid)
        self.assertTrue(any(i.code == "above_max" for i in r.issues))

    def test_apply_bridge_preserves_user_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cfg = {
            "column_mapping": {"Incident Angle": "incident_angle"},
            "parameter_constraints": {
                "incident_angle": {
                    "type": "continuous",
                    "min_value": 1.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "allowed_values": None,
                    "is_enabled": True,
                    "is_coupled": False,
                    "coupling_group": None,
                }
            },
        }
        apply_parameter_registry_to_config(project_root, cfg)
        self.assertEqual(cfg["parameter_constraints"]["incident_angle"]["min_value"], 1.0)
        self.assertEqual(cfg["parameter_constraints"]["incident_angle"]["max_value"], 5.0)
        reg = cfg.get("_parameter_registry")
        self.assertIsNotNone(reg)
        assert isinstance(reg, ParameterRegistry)
        self.assertIsNotNone(reg.get_by_alias_header("Incident Angle"))

    def test_legacy_entry_integer_from_discrete_step(self) -> None:
        d = ParameterDefinition(
            canonical_name="rotations",
            display_name="Rotations",
            data_type=ParameterDataType.DISCRETE_STEP,
            step=1.0,
        )
        entry = parameter_definition_to_legacy_entry(d)
        self.assertEqual(entry["type"], "integer")


if __name__ == "__main__":
    unittest.main()
