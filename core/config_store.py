"""
Config load/merge/save utilities.

We keep `config/default_config.json` as the base config, and persist user edits
to `config/user_config.json` so upgrades to the default remain possible.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = deepcopy(v)
    return out


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_default_config_path(project_root: Path) -> Path:
    return project_root / "config" / "default_config.json"


def get_user_config_path(project_root: Path) -> Path:
    return project_root / "config" / "user_config.json"


def get_default_parameter_registry_path(project_root: Path) -> Path:
    return project_root / "config" / "default_parameter_registry.json"


def get_user_parameter_registry_path(project_root: Path) -> Path:
    return project_root / "config" / "user_parameter_registry.json"


def load_effective_config(project_root: Path) -> dict[str, Any]:
    base = load_json(get_default_config_path(project_root))
    user = load_json(get_user_config_path(project_root))
    return deep_merge(base, user)


def load_user_config(project_root: Path) -> dict[str, Any]:
    return load_json(get_user_config_path(project_root))


def save_user_config(project_root: Path, user_config: dict[str, Any]) -> None:
    save_json(get_user_config_path(project_root), user_config)

