#!/usr/bin/env python3
"""
Ponto de entrada único para o fluxo do notebook flutter_benchmarks.ipynb:

  python run.py convert              # export TFLite (config.paths + config.convert)
  python run.py infer-coco           # inferência PC + COCOeval (GT JSON COCO)
  python run.py infer-yolo-gt        # inferência PC com labels YOLO (.txt)
  python run.py eval-mobile          # métricas de um JSON exportado pelo app
  python run.py plots                # compara vários JSONs mobile + gráficos
  python run.py list-models          # lista modelos do models.json do Flutter

Variável de ambiente opcional: BENCHMARKS_PC_CONFIG=/caminho/config.yaml

Código do pacote: src/benchmarks_pc/. Na raiz, `run.py` só reexporta este módulo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from benchmarks_pc.settings import load_config


def _paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("paths") or {}


def _infer(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("inference") or {}


def cmd_convert(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.convert_model import export_from_config

    export_from_config(cfg)


def _resolved_model_path(cfg: Dict[str, Any]) -> str:
    from benchmarks_pc.models_manifest import resolve_model_for_inference

    return str(resolve_model_for_inference(cfg))


def _model_for_infer_coco(cfg: Dict[str, Any]) -> str:
    from benchmarks_pc.models_manifest import resolve_model_path_string

    p = _paths(cfg)
    raw = p.get("model_for_infer_coco") or p.get("model_for_coco")
    if raw is not None and str(raw).strip() != "":
        return str(resolve_model_path_string(cfg, str(raw)))
    return _resolved_model_path(cfg)


def _model_for_infer_epi(cfg: Dict[str, Any]) -> str:
    from benchmarks_pc.models_manifest import resolve_model_path_string

    p = _paths(cfg)
    raw = p.get("model_for_infer_epi") or p.get("model_for_infer_yolo_gt") or p.get("model_for_yolo_gt")
    if raw is not None and str(raw).strip() != "":
        return str(resolve_model_path_string(cfg, str(raw)))
    return _resolved_model_path(cfg)


def cmd_infer_coco(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.inference_pc import inference_and_evaluate_coco_gt

    p = _paths(cfg)
    inf = _infer(cfg)
    out = Path(p.get("output_dir", "./benchmark_outputs"))
    out.mkdir(parents=True, exist_ok=True)
    pred_name = p.get("predictions_json", "predictions_coco.json")
    pred_path = out / pred_name if not Path(str(pred_name)).is_absolute() else Path(pred_name)

    class_id = inf.get("class_id_to_category_id")
    class_name = inf.get("class_name_to_category_id")
    mode = (inf.get("class_mapping_mode") or "auto").strip().lower()

    inference_and_evaluate_coco_gt(
        model_path=_model_for_infer_coco(cfg),
        val_images_dir=p["val_images_dir"],
        ann_file=p["coco_instances_json"],
        out_predictions_json=pred_path,
        imgsz=int(inf.get("imgsz", 640)),
        conf_thres=float(inf.get("conf", 0.001)),
        iou_nms=float(inf.get("iou_nms", 0.7)),
        device=inf.get("device"),
        class_id_to_category_id=class_id,
        class_name_to_category_id=class_name,
        class_mapping_mode=mode,
    )


def cmd_infer_yolo_gt(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.inference_pc import evaluate_yolo_with_yolo_gt

    p = _paths(cfg)
    inf = _infer(cfg)
    out = Path(p.get("output_dir", "./benchmark_outputs")) / "yolo_gt_eval"
    me = cfg.get("mobile_eval") or {}
    evaluate_yolo_with_yolo_gt(
        model_path=_model_for_infer_epi(cfg),
        images_dir=p["yolo_val_images"],
        labels_dir=p["yolo_val_labels"],
        output_dir=out,
        imgsz=int(inf.get("imgsz", 640)),
        conf_thres=float(inf.get("conf", 0.001)),
        iou_nms=float(inf.get("iou_nms", 0.7)),
        device=inf.get("device"),
        gt_class_names=me.get("yolo_class_names"),
        class_id_to_gt_class_id=inf.get("class_id_to_gt_class_id"),
        class_name_to_gt_class_id=inf.get("class_name_to_gt_class_id"),
    )


def cmd_eval_mobile(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.metrics_mobile import evaluate_mobile_predictions, evaluate_mobile_predictions_yolo
    from benchmarks_pc.mobile_pred_manifest import resolve_mobile_pred_path_string

    p = _paths(cfg)
    me = cfg.get("mobile_eval") or {}
    pred = p.get("mobile_pred_json")
    if not pred:
        raise SystemExit("Defina paths.mobile_pred_json no config.yaml")
    pred_path = resolve_mobile_pred_path_string(cfg, str(pred))
    gt_kind = (me.get("gt_kind") or "coco").strip().lower()
    aliases = me.get("class_name_aliases") or {}
    name_to_cat = me.get("class_name_to_category_id")
    iou_aux = float(me.get("iou_thr_aux", 0.5))

    if gt_kind == "yolo":
        result = evaluate_mobile_predictions_yolo(
            gt_images_dir=p["mobile_gt_yolo_images"],
            gt_labels_dir=p["mobile_gt_yolo_labels"],
            pred_json_path=str(pred_path),
            class_names=me.get("yolo_class_names"),
            dataset_yaml_path=p.get("dataset_yaml") if me.get("yolo_class_names") is None else None,
            class_name_aliases=aliases,
            class_name_to_category_id=name_to_cat,
            iou_thr_aux=iou_aux,
        )
    else:
        gt_json = p.get("mobile_gt_coco_json")
        if not gt_json:
            raise SystemExit("Para gt_kind=coco defina paths.mobile_gt_coco_json")
        result = evaluate_mobile_predictions(
            gt_json_path=gt_json,
            pred_json_path=str(pred_path),
            class_name_aliases=aliases,
            class_name_to_category_id=name_to_cat,
            iou_thr_aux=iou_aux,
        )

    out = Path(p.get("output_dir", "./benchmark_outputs"))
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "eval_mobile_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em: {report_path}")


def cmd_plots(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.mobile_pred_manifest import apply_mobile_predictions_manifest
    from benchmarks_pc.models_manifest import apply_manifest_plot_labels
    from benchmarks_pc.plots import run_mobile_comparison_from_config

    apply_mobile_predictions_manifest(cfg)
    apply_manifest_plot_labels(cfg)
    run_mobile_comparison_from_config(cfg)


def cmd_list_models(cfg: Dict[str, Any]) -> None:
    from benchmarks_pc.models_manifest import format_models_table

    print(format_models_table(cfg))


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmarks PC (YOLO / TFLite / COCO)")
    parser.add_argument("--config", type=str, default=None, help="Caminho para config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("convert", help="Exportar modelo (ex.: TFLite)")
    sub.add_parser("infer-coco", help="Inferência + métricas com annotations COCO JSON")
    sub.add_parser("infer-yolo-gt", help="Inferência + métricas com GT em labels YOLO")
    sub.add_parser("eval-mobile", help="Avaliar um JSON de predições do app")
    sub.add_parser("plots", help="Comparar vários JSONs mobile e gerar gráficos")
    sub.add_parser("list-models", help="Listar modelos do models.json (assets Flutter)")

    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    if args.command == "convert":
        cmd_convert(cfg)
    elif args.command == "infer-coco":
        cmd_infer_coco(cfg)
    elif args.command == "infer-yolo-gt":
        cmd_infer_yolo_gt(cfg)
    elif args.command == "eval-mobile":
        cmd_eval_mobile(cfg)
    elif args.command == "plots":
        cmd_plots(cfg)
    elif args.command == "list-models":
        cmd_list_models(cfg)
    else:
        parser.error("Comando desconhecido")


if __name__ == "__main__":
    main()
