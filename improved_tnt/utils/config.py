from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PATH_KEYS = {
    "cache_path",
    "checkpoint",
    "data_root",
    "interaction_root",
    "map_root",
    "output_root",
    "run_dir",
    "save_dir",
    "save_path",
    "map_cache_path",
    "val_cache_path",
    "val_map_cache_path",
}

PATH_LIST_KEYS = {"cache_paths"}


def _convert_paths(values: dict[str, Any]) -> dict[str, Any]:
    converted = dict(values)
    for key in PATH_KEYS:
        if converted.get(key) is not None:
            if isinstance(converted[key], list):
                raise TypeError(
                    f"Config key '{key}' expects a single path, but got a list. "
                    "Use 'cache_paths' for multiple cache files."
                )
            converted[key] = Path(converted[key])
    for key in PATH_LIST_KEYS:
        if converted.get(key) is not None:
            converted[key] = [Path(path) for path in converted[key]]
    return converted


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    config = _convert_paths(config)
    if config.get("val_cases") is not None:
        config["val_cases"] = [_convert_paths(case) for case in config["val_cases"]]
    return config


def config_namespace(path: str | Path) -> argparse.Namespace:
    return argparse.Namespace(**load_yaml_config(path))
