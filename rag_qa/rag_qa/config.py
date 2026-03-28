from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or (project_root() / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    p = cfg.get(key, "")
    path = Path(p)
    if not path.is_absolute():
        path = project_root() / path
    return path
