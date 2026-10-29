from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    includes = list(cfg.pop("includes", []))
    merged = OmegaConf.create()
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = OmegaConf.merge(merged, load_config(include_path))
    merged = OmegaConf.merge(merged, cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return merged


def config_container(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("root config must be a mapping")
    return value


def config_hash(cfg: DictConfig) -> str:
    payload = json.dumps(config_container(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_resolved_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)

