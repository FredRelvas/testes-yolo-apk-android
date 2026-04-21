"""Caminhos fixos do subprojeto benchmarks-pc e do repositório git."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Pasta `benchmarks-pc/` (config.yaml, predictions/, benchmark_outputs/)."""
    return Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    """Raiz do repositório `testes-yolo-apk-android/` (pai de benchmarks-pc/)."""
    return project_root().parent
