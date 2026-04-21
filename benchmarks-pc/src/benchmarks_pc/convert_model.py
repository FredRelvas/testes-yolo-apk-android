"""Exporta modelo Ultralytics (.pt) para TFLite (e variantes)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from benchmarks_pc.settings import load_config


def export_from_config(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {})
    convert = cfg.get("convert", {})
    model_path = paths.get("model_to_convert")
    if not model_path:
        raise ValueError("config: paths.model_to_convert é obrigatório para conversão.")
    fmt = convert.get("format", "tflite")
    imgsz = int(convert.get("imgsz", 640))
    half = bool(convert.get("half", False))
    int8 = bool(convert.get("int8", False))
    int8_data = convert.get("int8_data_yaml")

    out_dir = Path(paths.get("output_dir", "./benchmark_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    kwargs: Dict[str, Any] = {"format": fmt, "imgsz": imgsz}
    if half:
        kwargs["half"] = True
    if int8:
        kwargs["int8"] = True
        if int8_data:
            kwargs["data"] = str(int8_data)

    print(f"Exportando {model_path} com {kwargs} ...")
    paths_out = model.export(**kwargs)
    exported = Path(paths_out)
    print(f"Arquivo gerado: {exported}")
    return exported


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Conversão de modelo (ex.: TFLite)")
    p.add_argument("--config", type=str, default=None, help="YAML de configuração")
    args = p.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    export_from_config(cfg)


if __name__ == "__main__":
    main()
