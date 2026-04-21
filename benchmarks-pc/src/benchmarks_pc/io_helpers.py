"""Utilitários de caminho (ex.: localizar val2017 dentro de um dataset baixado)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_path(root: Path | str, target_name: str) -> Optional[Path]:
    root = Path(root)
    matches = list(root.rglob(target_name))
    return matches[0] if matches else None
