#!/usr/bin/env python3
"""Entrada na raiz de benchmarks-pc: adiciona `src/` ao path e delega ao pacote."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from benchmarks_pc.run import main

if __name__ == "__main__":
    main()
