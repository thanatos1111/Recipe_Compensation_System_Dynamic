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


def get_bundle_catalog(*, include_presets: bool = True) -> dict[str, dict[str, Any]]:
    """
    Return a simple catalog of known bundles.

    Shape is intentionally JSON-serializable to support future persistence.
    """
    out: dict[str, dict[str, Any]] = {}
    if include_presets:
        for name, targets in (MODEL_BUNDLE_PRESETS or {}).items():
            out[str(name)] = {"source": "preset", "models": dict(targets)}
    return out


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

