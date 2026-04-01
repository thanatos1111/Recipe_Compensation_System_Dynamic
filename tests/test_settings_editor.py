"""Tests for settings editor helpers (aliases, validation)."""

from __future__ import annotations

import unittest

from core.parameter_registry import (
    ParameterAlias,
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
)
from core.settings_editor import (
    apply_alias_pairs_to_registry,
    collect_alias_pairs,
    validate_alias_pairs,
    validate_duplicate_canonicals,
    validate_minimum_steps_dict,
)


class TestSettingsEditor(unittest.TestCase):
    def test_collect_and_apply_aliases(self) -> None:
        reg = ParameterRegistry()
        reg.add(
            ParameterDefinition(
                canonical_name="a",
                display_name="A",
                data_type=ParameterDataType.CONTINUOUS,
                aliases=[ParameterAlias(header="Col A")],
            )
        )
        pairs = collect_alias_pairs(reg)
        self.assertIn(("Col A", "a"), pairs)

        reg2 = ParameterRegistry()
        reg2.add(
            ParameterDefinition(
                canonical_name="a",
                display_name="A",
                data_type=ParameterDataType.CONTINUOUS,
                aliases=[],
            )
        )
        apply_alias_pairs_to_registry(reg2, [("H1", "a"), ("H2", "a")])
        self.assertEqual(len(reg2.get_by_canonical("a").aliases), 2)

    def test_validate_alias_pairs_unknown_canonical(self) -> None:
        errs = validate_alias_pairs([("X", "missing")], {"a"})
        self.assertTrue(any("unknown" in e.lower() for e in errs))

    def test_validate_duplicate_canonicals(self) -> None:
        errs = validate_duplicate_canonicals(["x", "x"])
        self.assertTrue(errs)

    def test_validate_minimum_steps(self) -> None:
        errs = validate_minimum_steps_dict({"p": -1.0})
        self.assertTrue(errs)


if __name__ == "__main__":
    unittest.main()
