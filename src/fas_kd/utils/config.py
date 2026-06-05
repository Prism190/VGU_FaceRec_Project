from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def parse_override_value(raw_value: str) -> Any:
    # YAML parsing keeps booleans/numbers/lists ergonomic for CLI overrides.
    return yaml.safe_load(raw_value)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node: dict[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            node[key] = {}
            child = node[key]
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set nested key under non-dict node: {dotted_key}")
        node = child
    node[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value form: {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(config, key.strip(), parse_override_value(value.strip()))
    return config


def ensure_runtime_dirs(config: dict[str, Any]) -> None:
    output_root = Path(config["experiment"]["output_root"])
    for name in ("checkpoints", "logs", "work"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
