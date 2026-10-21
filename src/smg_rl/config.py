"""Configuration loading with explicit project-relative paths."""

import os
from pathlib import Path

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    for key, value in config["paths"].items():
        expanded = Path(os.path.expandvars(os.path.expanduser(value)))
        config["paths"][key] = str(expanded if expanded.is_absolute() else root / expanded)
    config["root"] = str(root)
    return config
