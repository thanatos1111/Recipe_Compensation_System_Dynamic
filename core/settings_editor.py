"""
Pure helpers for the tabbed Settings dialog (alias merging, validation).

Keeps Qt out of this module so logic can be unit-tested.
"""

from __future__ import annotations

from typing import Optional

from core.parameter_registry import (
    ParameterAlias,
    ParameterDataType,
    ParameterDefinition,
    ParameterRegistry,
    validate_registry,
)


def collect_alias_pairs(reg: ParameterRegistry) -> list[tuple[str, str]]:
    """Return (Excel header, canonical_name) pairs for the alias editor, sorted for display."""
    pairs: list[tuple[str, str]] = []
    for d in reg._by_canonical.values():
        for a in d.aliases:
            pairs.append((a.header, d.canonical_name))
    pairs.sort(key=lambda t: (t[1].lower(), t[0].lower()))
    return pairs


def apply_alias_pairs_to_registry(reg: ParameterRegistry, pairs: list[tuple[str, str]]) -> None:
    """
    Replace alias lists on each definition from flat (header, canonical) rows.

    Multiple headers may map to the same canonical. Duplicate (header, canonical)
    pairs are skipped.
    """
    for d in reg._by_canonical.values():
        d.aliases = []
    seen: set[tuple[str, str]] = set()
    for header, canonical in pairs:
        h = (header or "").strip()
        c = (canonical or "").strip()
        if not h or not c:
            continue
        key = (h.lower(), c)
        if key in seen:
            continue
        seen.add(key)
        d = reg.get_by_canonical(c)
        if d is None:
            continue
        d.aliases.append(ParameterAlias(header=h))
    reg.rebuild_indexes()


def validate_alias_pairs(
    pairs: list[tuple[str, str]],
    canonical_names: set[str],
) -> list[str]:
    """Return human-readable validation errors for alias rows."""
    errors: list[str] = []
    for i, (header, canonical) in enumerate(pairs):
        row = i + 1
        if not (header or "").strip():
            errors.append(f"Alias row {row}: Excel header is empty.")
        c = (canonical or "").strip()
        if not c:
            errors.append(f"Alias row {row}: canonical parameter is empty.")
        elif c not in canonical_names:
            errors.append(f"Alias row {row}: unknown canonical parameter {c!r}.")
    return errors


def validate_minimum_steps_dict(steps: dict[str, float]) -> list[str]:
    errors: list[str] = []
    for k, v in steps.items():
        name = str(k).strip()
        if not name:
            errors.append("Minimum steps: empty parameter name.")
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"Minimum steps ({name}): value is not a number.")
            continue
        if fv < 0:
            errors.append(f"Minimum steps ({name}): value must be non-negative.")
    return errors


def validate_duplicate_canonicals(names: list[str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for n in names:
        c = (n or "").strip()
        if not c:
            continue
        if c in seen:
            errors.append(f"Duplicate canonical parameter name: {c!r}.")
        seen.add(c)
    return errors


def merge_registry_validation_messages(reg: ParameterRegistry) -> list[str]:
    """Turn registry validation issues into display strings."""
    result = validate_registry(reg)
    if result.valid:
        return []
    return [f"{i.field or 'parameter'}: {i.message}" for i in result.issues]


def parse_optional_float(text: str) -> Optional[float]:
    s = (text or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Not a valid number: {text!r}") from None


def data_type_from_combo_text(text: str) -> ParameterDataType:
    s = (text or "").strip().lower()
    mapping = {
        "continuous": ParameterDataType.CONTINUOUS,
        "discrete_step": ParameterDataType.DISCRETE_STEP,
        "categorical": ParameterDataType.CATEGORICAL,
        "boolean": ParameterDataType.BOOLEAN,
    }
    return mapping.get(s, ParameterDataType.CONTINUOUS)


def combo_text_from_data_type(dt: ParameterDataType) -> str:
    return dt.value
