"""
Scoped parameter and D.Spec profiles by material name and target position (1–12).

Precedence (later merges win within the same layer; see ``matching_profiles_ordered``):
- Global parameter registry / default_config spec_settings
- Wildcard material (\"\" ) + target_position presets
- Material-only profiles
- Material + target_position profiles

Legacy ``material_overrides`` in user_config is still supported, but scoped spec profiles take precedence.
"""

from __future__ import annotations

import fnmatch
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from core.config_store import deep_merge, load_json, load_user_config
from core.parameter_registry import build_effective_registry, load_effective_parameter_registry

SCOPED_SETTINGS_VERSION = 1
TARGET_POSITION_MIN = 1
TARGET_POSITION_MAX = 12

# Incident-angle dependent rules by preset group (driver: linear_offset).
_RULES_A = [
    {"driver_min": -190.0, "driver_max": -175.0, "min_value": 0.0, "max_value": 46.0},
    {"driver_min": -175.0, "driver_max": 15.0, "min_value": 0.0, "max_value": 50.0},
]
_RULES_B = [
    {"driver_min": 175.0, "driver_max": 190.0, "min_value": 0.0, "max_value": 46.0},
    {"driver_min": -15.0, "driver_max": 175.0, "min_value": 0.0, "max_value": 50.0},
]
_RULES_C = [
    {"driver_min": -190.0, "driver_max": -175.0, "min_value": 18.0, "max_value": 78.0},
    {"driver_min": -175.0, "driver_max": 15.0, "min_value": 14.0, "max_value": 78.0},
]
_RULES_D = [
    {"driver_min": 175.0, "driver_max": 190.0, "min_value": 18.0, "max_value": 78.0},
    {"driver_min": -15.0, "driver_max": 175.0, "min_value": 14.0, "max_value": 78.0},
]

# Prompt 2.1 grouping for incident-angle presets.
_INCIDENT_GROUP: dict[int, str] = {
    1: "A",
    2: "C",
    3: "A",
    4: "C",
    5: "B",
    6: "D",
    7: "A",
    8: "C",
    9: "A",
    10: "C",
    11: "B",
    12: "D",
}

_RULES_BY_GROUP = {"A": _RULES_A, "B": _RULES_B, "C": _RULES_C, "D": _RULES_D}


def _linear_offset_range_for_target_position(tp: int) -> tuple[float, float]:
    if tp in (1, 2, 3, 4, 7, 8, 9, 10):
        return -190.0, 15.0
    return -15.0, 190.0


def default_parameter_preset_profile(target_position: int) -> dict[str, Any]:
    """Single wildcard (any material) preset for one target position."""
    g = _INCIDENT_GROUP.get(target_position, "A")
    lo_min, lo_max = _linear_offset_range_for_target_position(target_position)
    rules = _RULES_BY_GROUP[g]
    return {
        "id": f"preset-tp-{target_position}",
        "enabled": True,
        "material_name": "",
        "target_position": target_position,
        "notes": "Default target-position preset (Prompt 2.1)",
        "priority": 0,
        "parameter_overrides": {
            "linear_offset": {
                "type": "continuous",
                "min_value": lo_min,
                "max_value": lo_max,
                "step": 0.0,
                "is_enabled": True,
            },
            "incident_angle": {
                "dependent_constraints": [
                    {
                        "driver_parameter": "linear_offset",
                        "rules": list(rules),
                    }
                ],
            },
        },
    }


def default_scoped_settings_dict() -> dict[str, Any]:
    """Factory for shipped defaults (also used if JSON file is missing)."""
    return {
        "version": SCOPED_SETTINGS_VERSION,
        "parameter_profiles": [default_parameter_preset_profile(tp) for tp in range(1, 13)],
        "spec_profiles": [],
    }


def get_default_scoped_settings_path(project_root: Path) -> Path:
    return project_root / "config" / "default_scoped_settings.json"


def load_effective_scoped_settings(project_root: Path, user_config: dict[str, Any]) -> dict[str, Any]:
    """Merge default file with ``user_config['scoped_settings']`` (user wins)."""
    path = get_default_scoped_settings_path(project_root)
    base = load_json(path)
    if not base:
        base = default_scoped_settings_dict()
    user = user_config.get("scoped_settings")
    if isinstance(user, dict) and user:
        merged = deep_merge(base, user)
        # If user saved an empty profile list, keep shipped target-position presets.
        if not merged.get("parameter_profiles"):
            merged["parameter_profiles"] = list(base.get("parameter_profiles") or [])
        return merged
    return base


def target_position_from_target_id(material_name: str, target_id: str) -> Optional[int]:
    """
    Parse target position 1–12 from ``target_id``.

    Preferred: ``<material>.<n>`` (recommended format). Also accepts trailing ``.<n>``.
    """
    raw = (target_id or "").strip()
    if not raw:
        return None
    m = re.match(rf"^{re.escape(material_name)}\.(\d+)$", raw, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if TARGET_POSITION_MIN <= n <= TARGET_POSITION_MAX:
            return n
    m2 = re.search(r"\.(\d+)\s*$", raw)
    if m2:
        n = int(m2.group(1))
        if TARGET_POSITION_MIN <= n <= TARGET_POSITION_MAX:
            return n
    if re.match(r"^\d{1,2}$", raw.strip()):
        n = int(raw.strip())
        if TARGET_POSITION_MIN <= n <= TARGET_POSITION_MAX:
            return n
    return None


def _material_matches(profile_material: str, material_name: str) -> bool:
    pm = (profile_material or "").strip()
    if not pm:
        return True
    pm_l = pm.lower()
    mn_l = (material_name or "").strip().lower()
    if any(ch in pm for ch in ("*", "?")):
        # Allow simple globbing for sheet/material names (case-insensitive).
        return fnmatch.fnmatchcase(mn_l, pm_l)
    return pm_l == mn_l


def _profile_layer(p: dict[str, Any]) -> int:
    """Higher layer = merged later (more specific)."""
    has_m = bool((p.get("material_name") or "").strip())
    tp = p.get("target_position")
    has_tp = tp is not None and str(tp).strip() != ""
    if has_m and has_tp:
        return 3
    if has_m and not has_tp:
        return 2
    if not has_m and has_tp:
        return 1
    return 0


def matching_parameter_profiles_ordered(
    scoped: dict[str, Any],
    material_name: str,
    target_position: Optional[int],
) -> list[dict[str, Any]]:
    """Return enabled profiles that match, in merge order (least specific first)."""
    profiles = scoped.get("parameter_profiles")
    if not isinstance(profiles, list):
        return []
    matched: list[dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict) or not p.get("enabled", True):
            continue
        if not _material_matches(str(p.get("material_name", "")), material_name):
            continue
        tp_raw = p.get("target_position")
        has_tp = tp_raw is not None and str(tp_raw).strip() != ""
        if has_tp:
            try:
                tp_val = int(tp_raw)
            except (TypeError, ValueError):
                continue
            if target_position is None or tp_val != target_position:
                continue
        else:
            # Material-only profile: no target filter in profile
            pass
        matched.append(p)

    matched.sort(
        key=lambda x: (_profile_layer(x), int(x.get("priority", 0) or 0), str(x.get("id", "")))
    )
    return matched


def merge_legacy_constraint_entry(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Merge one parameter's legacy constraint dict; ``dependent_constraints`` replaces when provided."""
    out = dict(base)
    for k, v in over.items():
        if k == "dependent_constraints":
            out[k] = deepcopy(v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_parameter_constraints_delta(base_pc: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge per-parameter entries into a legacy parameter_constraints dict."""
    out: dict[str, Any] = deepcopy(base_pc)
    for pname, pover in (delta or {}).items():
        if not isinstance(pover, dict):
            continue
        if pname not in out or not isinstance(out.get(pname), dict):
            out[pname] = deepcopy(pover)
        else:
            out[pname] = merge_legacy_constraint_entry(out[pname], pover)
    return out


def resolve_parameter_override_delta(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``parameter_overrides`` from profiles in order (later wins per field via full merge)."""
    merged_delta: dict[str, Any] = {}
    for p in profiles:
        po = p.get("parameter_overrides")
        if not isinstance(po, dict):
            continue
        merged_delta = merge_parameter_constraints_delta(merged_delta, po)
    return merged_delta


def matching_spec_profiles_ordered(
    scoped: dict[str, Any],
    material_name: str,
    target_position: Optional[int],
) -> list[dict[str, Any]]:
    profiles = scoped.get("spec_profiles")
    if not isinstance(profiles, list):
        return []
    matched: list[dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict) or not p.get("enabled", True):
            continue
        if not _material_matches(str(p.get("material_name", "")), material_name):
            continue
        tp_raw = p.get("target_position")
        has_tp = tp_raw is not None and str(tp_raw).strip() != ""
        if has_tp:
            try:
                tp_val = int(tp_raw)
            except (TypeError, ValueError):
                continue
            if target_position is None or tp_val != target_position:
                continue
        matched.append(p)
    matched.sort(
        key=lambda x: (_profile_layer(x), int(x.get("priority", 0) or 0), str(x.get("id", "")))
    )
    return matched


def merge_spec_dicts(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Shallow+one-level merge for spec_settings-like flat dicts."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **deepcopy(v)}
        else:
            out[k] = deepcopy(v)
    return out


def resolve_spec_profile_fields(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for p in profiles:
        spec = p.get("spec")
        if isinstance(spec, dict):
            merged = merge_spec_dicts(merged, spec)
    return merged


def warn_ambiguous_dependent_rules(
    scoped: dict[str, Any],
    material_name: str,
    target_position: Optional[int],
) -> list[str]:
    """Detect overlapping driver spans for the same driver_parameter on incident_angle (same scope)."""
    warnings: list[str] = []
    profs = matching_parameter_profiles_ordered(scoped, material_name, target_position)
    for p in profs:
        po = p.get("parameter_overrides") or {}
        ia = po.get("incident_angle") if isinstance(po, dict) else None
        if not isinstance(ia, dict):
            continue
        dcs = ia.get("dependent_constraints")
        if not isinstance(dcs, list):
            continue
        for dc in dcs:
            if not isinstance(dc, dict):
                continue
            rules = dc.get("rules")
            if not isinstance(rules, list) or len(rules) < 2:
                continue
            driver = str(dc.get("driver_parameter", ""))
            # pairwise overlap check on [driver_min, driver_max]
            spans: list[tuple[float, float]] = []
            for r in rules:
                if not isinstance(r, dict):
                    continue
                a = r.get("driver_min")
                b = r.get("driver_max")
                if a is None or b is None:
                    continue
                lo, hi = float(a), float(b)
                if lo > hi:
                    lo, hi = hi, lo
                spans.append((lo, hi))
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    a1, a2 = spans[i]
                    b1, b2 = spans[j]
                    if max(a1, b1) <= min(a2, b2) + 1e-9:
                        pid = p.get("id", "?")
                        warnings.append(
                            f"Profile {pid}: overlapping {driver} spans may be ambiguous for incident_angle rules."
                        )
    return warnings


def apply_scoped_parameter_constraints_to_config(
    project_root: Path,
    config: dict[str, Any],
    *,
    material_name: str,
    target_id: str,
    scoped: Optional[dict[str, Any]] = None,
) -> None:
    """
    Merge scoped parameter overrides into ``config['parameter_constraints']`` and rebuild
    ``config['_parameter_registry']``.

    Call only after :func:`apply_parameter_registry_to_config`.
    """
    if scoped is None:
        scoped = load_effective_scoped_settings(project_root, load_user_config(project_root))

    tp = target_position_from_target_id(material_name, target_id)
    profiles = matching_parameter_profiles_ordered(scoped, material_name, tp)
    delta = resolve_parameter_override_delta(profiles)
    if not delta:
        column_mapping = config.get("column_mapping") or {}
        reg_disk = load_effective_parameter_registry(project_root)
        config["_parameter_registry"] = build_effective_registry(
            reg_disk, config.get("parameter_constraints") or {}, column_mapping
        )
        return

    base_pc = config.get("parameter_constraints") or {}
    config["parameter_constraints"] = merge_parameter_constraints_delta(base_pc, delta)
    column_mapping = config.get("column_mapping") or {}
    reg_disk = load_effective_parameter_registry(project_root)
    config["_parameter_registry"] = build_effective_registry(
        reg_disk, config["parameter_constraints"], column_mapping
    )


def resolve_effective_spec_dict(
    config: dict[str, Any],
    scoped: dict[str, Any],
    *,
    material_name: str,
    target_id: Optional[str],
) -> dict[str, Any]:
    """
    Merge global spec_settings with scoped spec profiles and legacy material_overrides.

    Order: global → legacy material → legacy target → scoped (wildcard/material/material+tp).

    Rationale: legacy overrides are kept for backward compatibility, but if the user
    defines scoped spec profiles (via the Settings dialog), those should take effect.
    """
    base = dict(config.get("spec_settings") or {})
    tp = target_position_from_target_id(material_name, target_id or "") if target_id else None

    mo = config.get("material_overrides", {}) or {}
    mats = mo.get("materials", {}) or {}
    block = mats.get(material_name, {}) or {}
    mat_spec = block.get("spec_settings")
    if isinstance(mat_spec, dict):
        base = merge_spec_dicts(base, mat_spec)

    if target_id:
        tspec = (block.get("targets") or {}).get(target_id, {}) or {}
        ts = tspec.get("spec_settings")
        if isinstance(ts, dict):
            base = merge_spec_dicts(base, ts)

    sprof = matching_spec_profiles_ordered(scoped, material_name, tp)
    base = merge_spec_dicts(base, resolve_spec_profile_fields(sprof))
    return base


def resolve_effective_max_lifetime(
    config: dict[str, Any],
    scoped: dict[str, Any],
    *,
    material_name: str,
    target_id: Optional[str],
) -> Optional[float]:
    """Scoped spec profile ``max_lifetime`` overrides legacy when present."""
    tp = target_position_from_target_id(material_name, target_id or "") if target_id else None
    profiles = matching_spec_profiles_ordered(scoped, material_name, tp)
    for p in reversed(profiles):
        ml = p.get("max_lifetime")
        if ml is not None:
            try:
                return float(ml)
            except (TypeError, ValueError):
                continue
    mo = config.get("material_overrides", {}) or {}
    block = (mo.get("materials", {}) or {}).get(material_name, {}) or {}
    if target_id:
        v = (block.get("targets", {}) or {}).get(target_id, {}) or {}
        if v.get("max_lifetime") is not None:
            return float(v["max_lifetime"])
    v2 = block.get("max_lifetime")
    return float(v2) if v2 is not None else None


_SHIPPED_TP_PRESET = re.compile(r"^preset-tp-(\d+)$")


def is_shipped_target_position_preset(p: dict[str, Any]) -> bool:
    """Built-in target-position presets (Prompt 2.1) — validated once in factory data, not edited in UI."""
    return bool(_SHIPPED_TP_PRESET.match(str(p.get("id", "")).strip()))


def validate_scoped_parameter_profile(p: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if is_shipped_target_position_preset(p):
        return []
    mid = str(p.get("id", "")).strip()
    if not mid:
        errs.append("Profile id is required.")
    mat = str(p.get("material_name", "") or "").strip()
    tp = p.get("target_position")
    if tp is not None and str(tp).strip() != "":
        try:
            tpi = int(tp)
            if tpi < TARGET_POSITION_MIN or tpi > TARGET_POSITION_MAX:
                errs.append(f"target_position must be {TARGET_POSITION_MIN}–{TARGET_POSITION_MAX}.")
        except (TypeError, ValueError):
            errs.append("target_position must be an integer.")
    if mat == "" and (tp is None or str(tp).strip() == ""):
        errs.append(
            "Set a material name and/or target position (1–12). "
            "Material may be blank when target position is set (wildcard row)."
        )
    return errs


def validate_scoped_spec_profile(p: dict[str, Any]) -> list[str]:
    errs = validate_scoped_parameter_profile(p)
    if not isinstance(p.get("spec"), dict):
        errs.append("Spec profile needs a spec object.")
    return errs
