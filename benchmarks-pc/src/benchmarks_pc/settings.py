"""Carrega config.yaml (YAML) para os scripts da pasta benchmarks-pc."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from benchmarks_pc.layout import project_root

_DEFAULT_NAME = "config.yaml"


def default_config_path() -> Path:
    base = project_root()
    env = os.environ.get("BENCHMARKS_PC_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return base / _DEFAULT_NAME


def load_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve() if path else default_config_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {p}\n"
            f"Copie config.example.yaml para {_DEFAULT_NAME} ou defina BENCHMARKS_PC_CONFIG."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("config.yaml deve ser um objeto YAML no topo (mapa ch/valor).")
    return data


def deep_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out
