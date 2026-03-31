from __future__ import annotations

import pytest

from core.bundles import (
    CUSTOM_MODEL_BUNDLES_KEY,
    describe_bundle,
    get_bundle_catalog,
    get_effective_model_bundles,
    parse_custom_bundles_from_config,
    validate_custom_bundle,
)


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


def test_validate_custom_bundle_rejects_preset_name_collision() -> None:
    with pytest.raises(ValueError):
        validate_custom_bundle(name="balanced_default", models={"rs": "gbr", "thickness": "gbr", "rsu": "gbr"})


def test_validate_custom_bundle_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError):
        validate_custom_bundle(name="my_bundle", models={"rs": "not_a_model", "thickness": "gbr", "rsu": "gbr"})


def test_parse_custom_bundles_from_config_skips_invalid_entries() -> None:
    config = {
        CUSTOM_MODEL_BUNDLES_KEY: {
            "": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"},
            "bad_missing_target": {"rs": "gbr", "thickness": "gbr"},
            "bad_unknown_model": {"rs": "nope", "thickness": "gbr", "rsu": "gbr"},
            "good_one": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"},
        }
    }
    parsed = parse_custom_bundles_from_config(config)
    assert "good_one" in parsed
    assert "bad_missing_target" not in parsed
    assert "bad_unknown_model" not in parsed


def test_get_bundle_catalog_merges_preset_and_custom_sources() -> None:
    config = {CUSTOM_MODEL_BUNDLES_KEY: {"my_custom": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"}}}
    catalog = get_bundle_catalog(config=config)
    assert catalog["balanced_default"]["source"] == "preset"
    assert catalog["my_custom"]["source"] == "custom"


def test_get_effective_model_bundles_includes_custom() -> None:
    config = {CUSTOM_MODEL_BUNDLES_KEY: {"my_custom": {"rs": "gbr", "thickness": "gbr", "rsu": "gbr"}}}
    bundles = get_effective_model_bundles(config)
    assert "balanced_default" in bundles
    assert "my_custom" in bundles
    assert bundles["my_custom"]["rs"] == "gbr"

