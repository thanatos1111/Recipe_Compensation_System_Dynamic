from __future__ import annotations

import pytest

from core.bundles import describe_bundle, get_bundle_catalog


def test_get_bundle_catalog_includes_presets() -> None:
    catalog = get_bundle_catalog()
    assert "balanced_default" in catalog
    assert catalog["balanced_default"]["source"] == "preset"
    assert catalog["balanced_default"]["models"]["rs"] == "gbr"


def test_describe_bundle_balanced_default_targets_present() -> None:
    desc = describe_bundle("balanced_default", catalog=get_bundle_catalog())
    by_target = desc.model_by_target()
    assert desc.bundle_name == "balanced_default"
    assert desc.source == "preset"
    assert by_target["rs"].model_name == "gbr"
    assert by_target["thickness"].model_name == "gbr"
    assert by_target["rsu"].model_name == "gbr"


def test_describe_bundle_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        describe_bundle("does_not_exist", catalog=get_bundle_catalog())

