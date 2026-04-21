"""Modelos TFLite listados no `models.json` dos assets do app Flutter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks_pc.layout import repo_root

# Lista canónica usada pelo exemplo Android (na raiz do repo git)
DEFAULT_MODELS_JSON = (
    repo_root() / "apk-yolo-flutter/example/android/app/src/main/assets/models.json"
)


def default_models_json_path() -> Path:
    return DEFAULT_MODELS_JSON


def effective_models_json(cfg: Dict[str, Any]) -> Path:
    """Caminho do manifest: `flutter_assets.models_json` ou o do exemplo Android no repo."""
    fa = cfg.get("flutter_assets") or {}
    if fa.get("models_json"):
        return Path(fa["models_json"]).expanduser().resolve()
    return DEFAULT_MODELS_JSON


def load_models_manifest(models_json: str | Path) -> List[Dict[str, Any]]:
    p = Path(models_json).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"models.json não encontrado: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("models.json deve ser um array JSON de objetos.")
    return data


def models_directory(cfg: Dict[str, Any]) -> Path:
    mj = effective_models_json(cfg)
    fa = cfg.get("flutter_assets") or {}
    if fa.get("models_dir"):
        return Path(fa["models_dir"]).expanduser().resolve()
    return mj.parent


def manifest_path(cfg: Dict[str, Any]) -> Path:
    return effective_models_json(cfg)


def iter_benchmark_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in entries:
        if e.get("benchmark", True) is not False:
            out.append(e)
    return out


def pick_default_entry(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    bench = iter_benchmark_entries(entries)
    for e in bench:
        if e.get("default") is True:
            return e
    if bench:
        return bench[0]
    if entries:
        return entries[0]
    raise ValueError("models.json não contém entradas de modelo.")


def resolve_tflite_path(models_dir: Path, filename: str) -> Path:
    return (models_dir / filename).resolve()


def _norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def resolve_model_path_string(cfg: Dict[str, Any], value: str) -> Path:
    """
    Resolve um modelo a partir de:
      - caminho absoluto ou relativo a um ficheiro existente;
      - \"label\" igual ao campo \"label\" (ou nome do \"file\") em models.json;
      - nome de ficheiro .tflite existente em flutter_assets.models_dir.
    """
    v = str(value).strip().strip('"').strip("'")
    if not v:
        raise ValueError("Valor de modelo vazio.")
    direct = Path(v).expanduser()
    if direct.is_file():
        return direct.resolve()

    mj_path = effective_models_json(cfg)
    models_dir = models_directory(cfg)
    if mj_path.is_file():
        entries = load_models_manifest(mj_path)
        nv = _norm_label(v)
        for e in entries:
            lab_raw = e.get("label") or ""
            fn = str(e.get("file") or "")
            if not fn:
                continue
            if nv == _norm_label(lab_raw) or nv == _norm_label(fn) or nv == _norm_label(Path(fn).name):
                out = resolve_tflite_path(models_dir, fn)
                if out.is_file():
                    return out.resolve()
                raise FileNotFoundError(
                    f"models.json referencia '{fn}' mas o ficheiro não existe em {models_dir}"
                )

    name = Path(v).name
    cand = resolve_tflite_path(models_dir, name)
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Modelo não resolvido: {value!r}. Indique um caminho, o \"label\" de models.json "
        f"({mj_path}), ou o nome do .tflite dentro de {models_dir}."
    )


def resolve_model_for_inference(cfg: Dict[str, Any]) -> Path:
    """
    Determina o ficheiro .tflite (ou outro modelo) para inferência.

    Prioridade:
      1) paths.model_for_inference — caminho, label do models.json, ou nome .tflite em models_dir
      2) Entrada com \"default\": true no models.json (ou primeira com benchmark)
    """
    p = cfg.get("paths") or {}
    explicit = p.get("model_for_inference")
    mj_path = effective_models_json(cfg)

    if explicit is not None and str(explicit).strip() != "":
        return resolve_model_path_string(cfg, str(explicit))

    if not mj_path.is_file():
        raise FileNotFoundError(
            "Defina paths.model_for_inference OU coloque models.json em:\n"
            f"  {DEFAULT_MODELS_JSON}\n"
            "ou configure flutter_assets.models_json no config.yaml."
        )

    entries = load_models_manifest(mj_path)
    models_dir = models_directory(cfg)
    entry = pick_default_entry(entries)
    fn = entry.get("file")
    if not fn:
        raise ValueError(f"Entrada sem 'file' no manifest: {entry}")
    out = resolve_tflite_path(models_dir, str(fn))
    if not out.is_file():
        raise FileNotFoundError(
            f"Modelo do manifest não encontrado no disco: {out}\n"
            f"Copie o ficheiro '{fn}' para {models_dir} (ver {mj_path})."
        )
    return out


def manifest_labels_for_benchmark_entries(cfg: Dict[str, Any]) -> List[str]:
    mj = manifest_path(cfg)
    if not mj.is_file():
        return []
    entries = iter_benchmark_entries(load_models_manifest(mj))
    return [str(e.get("label") or e.get("file", "")) for e in entries]


def apply_manifest_plot_labels(cfg: Dict[str, Any]) -> None:
    """Se configurado, preenche paths.model_display_names a partir do models.json."""
    fa = cfg.get("flutter_assets") or {}
    if not fa.get("use_manifest_labels_for_plots"):
        return
    paths = cfg.setdefault("paths", {})
    if paths.get("model_display_names"):
        return
    labels = manifest_labels_for_benchmark_entries(cfg)
    preds = paths.get("mobile_pred_files") or []
    if not labels or not preds:
        return
    if len(labels) != len(preds):
        print(
            f"Aviso: use_manifest_labels_for_plots — {len(labels)} labels no manifest vs "
            f"{len(preds)} ficheiros em mobile_pred_files; não alterando nomes."
        )
        return
    paths["model_display_names"] = labels


def format_models_table(cfg: Dict[str, Any]) -> str:
    mj = manifest_path(cfg)
    if not mj.is_file():
        return f"(models.json não encontrado: {mj})"
    entries = load_models_manifest(mj)
    fa = cfg.get("flutter_assets") or {}
    mdir = Path(fa["models_dir"]).expanduser().resolve() if fa.get("models_dir") else mj.parent
    lines = [f"models_json: {mj}", f"models_dir:  {mdir}", ""]
    for e in entries:
        fn = e.get("file", "?")
        exists = "sim" if (mdir / str(fn)).is_file() else "não"
        lines.append(
            f"  - {fn!s:45}  label={e.get('label', '')!s}  benchmark={e.get('benchmark', True)}  "
            f"default={e.get('default', False)}  no_disco={exists}"
        )
    return "\n".join(lines)
