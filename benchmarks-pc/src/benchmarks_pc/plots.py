"""Gráficos de comparação (mAP50, tempo de inferência) a partir de JSONs mobile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from benchmarks_pc.metrics_mobile import evaluate_mobile_predictions, evaluate_mobile_predictions_yolo
from benchmarks_pc.settings import load_config


def plot_comparison_bars(
    rows: List[Dict[str, Any]],
    save: bool,
    output_dir: Path,
    map50_filename: str = "comparison_map50.png",
    time_filename: str = "comparison_inference_time_ms.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_models = sorted({row["model"] for row in rows})
    cmap = plt.get_cmap("tab10")
    color_map = {model: cmap(i % 10) for i, model in enumerate(unique_models)}

    rows_map50 = [r for r in rows if r.get("mAP50") is not None]
    if rows_map50:
        model_names = [r["model"] for r in rows_map50]
        map50_values = [float(r["mAP50"]) for r in rows_map50]
        colors = [color_map[m] for m in model_names]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(model_names, map50_values, color=colors)
        plt.title("Comparação de mAP50")
        plt.xlabel("Modelo")
        plt.ylabel("mAP50")
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        if save:
            out_path = output_dir / map50_filename
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Gráfico salvo em: {out_path}")
        plt.close()
    else:
        print("Nenhum valor de mAP50 disponível para plotar.")

    rows_time = [r for r in rows if r.get("avg_inference_time_ms") is not None]
    if rows_time:
        model_names = [r["model"] for r in rows_time]
        time_values = [float(r["avg_inference_time_ms"]) for r in rows_time]
        colors = [color_map[m] for m in model_names]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(model_names, time_values, color=colors)
        plt.title("Comparação do tempo médio de inferência")
        plt.xlabel("Modelo")
        plt.ylabel("Tempo médio de inferência (ms)")
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom")
        plt.tight_layout()
        if save:
            out_path = output_dir / time_filename
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Gráfico salvo em: {out_path}")
        plt.close()
    else:
        print("Nenhum valor de tempo disponível para plotar.")


def run_mobile_comparison_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from benchmarks_pc.mobile_pred_manifest import resolve_mobile_pred_path_string

    paths = cfg.get("paths", {})
    pred_files = paths.get("mobile_pred_files") or []
    if not pred_files:
        raise ValueError("config: paths.mobile_pred_files deve listar os JSONs de predição mobile.")
    resolved_preds: List[str] = []
    for pf in pred_files:
        p = Path(str(pf))
        if p.is_file():
            resolved_preds.append(str(p.resolve()))
        else:
            resolved_preds.append(str(resolve_mobile_pred_path_string(cfg, str(pf))))
    pred_files = resolved_preds

    aliases = cfg.get("mobile_eval", {}).get("class_name_aliases") or {}
    name_to_cat = cfg.get("mobile_eval", {}).get("class_name_to_category_id")
    iou_aux = float(cfg.get("mobile_eval", {}).get("iou_thr_aux", 0.5))
    gt_kind = (cfg.get("mobile_eval", {}) or {}).get("gt_kind", "coco")

    gt_coco = paths.get("mobile_gt_coco_json") or cfg.get("plots", {}).get("gt_json_for_mobile_plots")
    yolo_imgs = paths.get("mobile_gt_yolo_images")
    yolo_lbls = paths.get("mobile_gt_yolo_labels")
    dataset_yaml = paths.get("dataset_yaml")
    yolo_class_names = cfg.get("mobile_eval", {}).get("yolo_class_names")

    display_names: List[str] = list(paths.get("model_display_names") or [])
    out_dir = Path(paths.get("graphs_output_dir") or Path(paths.get("output_dir", ".")) / "graphs")
    save = bool(cfg.get("plots", {}).get("save", True))

    comparison_rows: List[Dict[str, Any]] = []
    for index, pred_file in enumerate(pred_files):
        print(f"\nAvaliando: {pred_file}")
        if gt_kind == "yolo":
            if not yolo_imgs or not yolo_lbls:
                raise ValueError("Para gt_kind=yolo, defina paths.mobile_gt_yolo_images e mobile_gt_yolo_labels.")
            eval_result = evaluate_mobile_predictions_yolo(
                gt_images_dir=yolo_imgs,
                gt_labels_dir=yolo_lbls,
                pred_json_path=pred_file,
                class_names=yolo_class_names,
                dataset_yaml_path=dataset_yaml if yolo_class_names is None else None,
                class_name_aliases=aliases,
                class_name_to_category_id=name_to_cat,
                iou_thr_aux=iou_aux,
            )
        else:
            if not gt_coco:
                raise ValueError("Para gt_kind=coco, defina paths.mobile_gt_coco_json (ou plots.gt_json_for_mobile_plots).")
            eval_result = evaluate_mobile_predictions(
                gt_json_path=gt_coco,
                pred_json_path=pred_file,
                class_name_aliases=aliases,
                class_name_to_category_id=name_to_cat,
                iou_thr_aux=iou_aux,
            )

        model_name = display_names[index] if index < len(display_names) else (
            eval_result.get("model") or Path(pred_file).stem
        )
        map50 = eval_result["coco_metrics"]["mAP50"]
        avg_time = eval_result["timing"]["avg_inference_time_ms"]
        comparison_rows.append(
            {
                "file": pred_file,
                "model": model_name,
                "mAP50": map50,
                "avg_inference_time_ms": avg_time,
                "full_result": eval_result,
            }
        )

    print("\n=== Resumo ===")
    for row in comparison_rows:
        t = row["avg_inference_time_ms"]
        ts = f"{t:.3f}" if t is not None else "n/a"
        print(f"Modelo: {row['model']}\n  mAP50: {row['mAP50']:.6f}\n  Avg inference time (ms): {ts}\n")

    plot_comparison_bars(comparison_rows, save=save, output_dir=out_dir)

    summary_output = [
        {"file": r["file"], "model": r["model"], "mAP50": r["mAP50"], "avg_inference_time_ms": r["avg_inference_time_ms"]}
        for r in comparison_rows
    ]
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\nResumo salvo em: {summary_path}")
    return comparison_rows


def main(argv: Optional[list] = None) -> None:
    from benchmarks_pc.mobile_pred_manifest import apply_mobile_predictions_manifest
    from benchmarks_pc.models_manifest import apply_manifest_plot_labels

    p = argparse.ArgumentParser(description="Gráficos de comparação de JSONs mobile")
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    apply_mobile_predictions_manifest(cfg)
    apply_manifest_plot_labels(cfg)
    run_mobile_comparison_from_config(cfg)


if __name__ == "__main__":
    main()
