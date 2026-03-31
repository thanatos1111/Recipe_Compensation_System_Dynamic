"""
Bundle catalog and description helpers for benchmark UI transparency.

This module is intentionally UI-agnostic (pure functions + dataclasses) so it can
be unit tested and reused by future bundle features (e.g. custom bundle
creation/persistence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from core.benchmarking import MODEL_BUNDLE_PRESETS
from core.model_registry import get_supported_model_specs

CUSTOM_MODEL_BUNDLES_KEY = "custom_model_bundles"
_BUNDLE_TARGET_KEYS: tuple[str, ...] = ("rs", "thickness", "rsu")


@dataclass(frozen=True)
class BundleModelSpec:
    """Resolved model spec for a single benchmark target (rs/thickness/rsu)."""

    target: str
    model_name: str
    display_name: str
    family: str


@dataclass(frozen=True)
class BundleDescription:
    """Human/UI-friendly description of a model bundle."""

    bundle_name: str
    source: str  # "preset" or "custom"
    models: tuple[BundleModelSpec, BundleModelSpec, BundleModelSpec]

    def model_by_target(self) -> dict[str, BundleModelSpec]:
        return {m.target: m for m in self.models}


def _supported_model_names() -> set[str]:
    return set(get_supported_model_specs().keys())


def normalize_bundle_models(models: Mapping[str, Any]) -> dict[str, str]:
    """
    Normalize a bundle model mapping to ``{"rs": ..., "thickness": ..., "rsu": ...}``.

    Raises
    ------
    ValueError
        If required targets are missing.
    TypeError
        If the input is not a mapping.
    """
    if not isinstance(models, Mapping):
        raise TypeError("Bundle models must be a mapping.")
    out: dict[str, str] = {}
    for k in _BUNDLE_TARGET_KEYS:
        v = models.get(k)
        s = str(v or "").strip()
        if not s:
            raise ValueError(f"Bundle models missing required target {k!r}.")
        out[k] = s
    return out


def validate_custom_bundle(
    *,
    name: str,
    models: Mapping[str, Any],
    preset_names: Optional[set[str]] = None,
    supported_model_names: Optional[set[str]] = None,
) -> dict[str, str]:
    """
    Validate a custom bundle definition and return normalized model mapping.

    Rules
    -----
    - Name must be non-empty.
    - Name must not collide with preset bundles (preset bundles are read-only).
    - Model names must exist in the registry.
    """
    bn = str(name or "").strip()
    if not bn:
        raise ValueError("Bundle name is required.")

    preset = preset_names or set((MODEL_BUNDLE_PRESETS or {}).keys())
    if bn in preset:
        raise ValueError(f"Bundle name {bn!r} collides with a preset bundle and cannot be reused.")

    normalized = normalize_bundle_models(models)
    supported = supported_model_names or _supported_model_names()
    for target, mn in normalized.items():
        if mn not in supported:
            raise ValueError(f"Unsupported model name for {target!r}: {mn!r}.")
    return normalized


def parse_custom_bundles_from_config(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """
    Parse custom bundles from an effective config mapping.

    This parser is tolerant: invalid entries are skipped so a partially-invalid
    user config doesn't break the UI.
    """
    raw = (config or {}).get(CUSTOM_MODEL_BUNDLES_KEY, {})
    if not isinstance(raw, Mapping):
        return {}
    supported = _supported_model_names()
    preset = set((MODEL_BUNDLE_PRESETS or {}).keys())
    out: dict[str, dict[str, str]] = {}
    for name, models in raw.items():
        try:
            bn = str(name or "").strip()
            if not bn or bn in preset:
                continue
            if not isinstance(models, Mapping):
                continue
            normalized = normalize_bundle_models(models)
            if any(mn not in supported for mn in normalized.values()):
                continue
            out[bn] = normalized
        except Exception:
            continue
    return out


def get_bundle_catalog(
    *,
    config: Optional[Mapping[str, Any]] = None,
    include_presets: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Return a simple catalog of known bundles.

    Shape is intentionally JSON-serializable to support future persistence.
    """
    out: dict[str, dict[str, Any]] = {}
    if include_presets:
        for name, targets in (MODEL_BUNDLE_PRESETS or {}).items():
            out[str(name)] = {"source": "preset", "models": dict(targets)}
    if config is not None:
        for name, models in parse_custom_bundles_from_config(config).items():
            if name in out:
                # Presets are authoritative; skip collisions at runtime.
                continue
            out[str(name)] = {"source": "custom", "models": dict(models)}
    return out


def get_effective_model_bundles(config: Optional[Mapping[str, Any]] = None) -> dict[str, dict[str, str]]:
    """
    Return bundle defs for benchmark/backtest runners.

    Presets are always included. Custom bundles from config are merged in, and
    preset-name collisions are ignored (presets win).
    """
    merged: dict[str, dict[str, str]] = {k: dict(v) for k, v in (MODEL_BUNDLE_PRESETS or {}).items()}
    if config is None:
        return merged
    for name, models in parse_custom_bundles_from_config(config).items():
        if name in merged:
            continue
        merged[name] = dict(models)
    return merged


def describe_bundle(
    bundle_name: str,
    *,
    catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> BundleDescription:
    """
    Describe a bundle by name.

    Parameters
    ----------
    bundle_name:
        Bundle key to describe.
    catalog:
        Optional catalog mapping, e.g. from :func:`get_bundle_catalog`.
        If omitted, presets are used.
    """
    bn = str(bundle_name or "").strip()
    if not bn:
        raise ValueError("bundle_name is required.")

    if catalog is None:
        catalog = get_bundle_catalog()

    entry = catalog.get(bn)
    if entry is None:
        raise KeyError(bn)

    source = str(entry.get("source") or "preset").strip() or "preset"
    targets = entry.get("models") or {}
    if not isinstance(targets, Mapping):
        raise TypeError("Bundle entry 'models' must be a mapping.")

    model_specs = get_supported_model_specs()

    def _resolve(target: str) -> BundleModelSpec:
        mn = str(targets.get(target) or "").strip()
        meta = model_specs.get(mn) or {}
        display = str(meta.get("display_name") or mn or "unknown")
        family = str(meta.get("family") or "unknown")
        return BundleModelSpec(target=target, model_name=mn, display_name=display, family=family)

    models = (_resolve("rs"), _resolve("thickness"), _resolve("rsu"))
    return BundleDescription(bundle_name=bn, source=source, models=models)

