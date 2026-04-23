"""Gráficos de comparação (mAP50, tempo de inferência) a partir de JSONs mobile."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from benchmarks_pc.inference_pc import evaluate_yolo_with_yolo_gt, inference_and_evaluate_coco_gt
from benchmarks_pc.metrics_mobile import evaluate_mobile_predictions, evaluate_mobile_predictions_yolo
from benchmarks_pc.models_manifest import resolve_model_path_string
from benchmarks_pc.settings import load_config


def _fps_from_ms(ms: Optional[float]) -> Optional[float]:
    if ms is None or ms <= 0:
        return None
    return 1000.0 / ms


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
        metric_specs = [
            ("avg_inference_time_ms", "Média"),
            ("median_inference_time_ms", "Mediana"),
            ("p95_inference_time_ms", "P95"),
            ("p99_inference_time_ms", "P99"),
            ("std_inference_time_ms", "Desvio padrão"),
        ]
        available_metrics = [
            (key, label) for key, label in metric_specs if any(r.get(key) is not None for r in rows_time)
        ]
        if not available_metrics:
            print("Nenhuma métrica de tempo disponível para plotar.")
            return

        x = list(range(len(model_names)))
        width = min(0.82 / len(available_metrics), 0.2)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (key, label) in enumerate(available_metrics):
            values = [float(r.get(key, 0.0) or 0.0) for r in rows_time]
            offset = (i - (len(available_metrics) - 1) / 2.0) * width
            bars = ax.bar([xx + offset for xx in x], values, width=width, label=label)
            for bar in bars:
                y = bar.get_height()
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_title("Comparação do tempo de inferência (ms): média, mediana, P95, P99 e desvio padrão")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Tempo de inferência (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        if save:
            out_path = output_dir / time_filename
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Gráfico salvo em: {out_path}")
        plt.close(fig)
    else:
        print("Nenhum valor de tempo disponível para plotar.")


def plot_fps_bars(
    rows: List[Dict[str, Any]],
    save: bool,
    output_dir: Path,
    filename: str = "comparison_fps_mean.png",
) -> None:
    rows_fps = [r for r in rows if _fps_from_ms(r.get("avg_inference_time_ms")) is not None]
    if not rows_fps:
        print("Nenhum valor de FPS médio disponível para plotar.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"] for r in rows_fps]
    fps_values = [float(_fps_from_ms(r.get("avg_inference_time_ms")) or 0.0) for r in rows_fps]
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(model_names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, fps_values, color=colors)
    ax.set_title("Comparação de FPS médio (mobile)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("FPS médio")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    if save:
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Gráfico salvo em: {out_path}")
    plt.close(fig)


def plot_mobile_vs_pc_inference_bars(
    rows: List[Dict[str, Any]],
    save: bool,
    output_dir: Path,
    filename: str = "comparison_inference_time_mobile_vs_pc.png",
    gt_kind: str = "coco",
) -> None:
    paired = [
        r
        for r in rows
        if r.get("avg_inference_time_ms") is not None and r.get("pc_avg_inference_time_ms") is not None
    ]
    if not paired:
        print("Nenhum par mobile/PC com tempo disponível para plotar.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"] for r in paired]
    mobile_times = [float(r["avg_inference_time_ms"]) for r in paired]
    pc_times = [float(r["pc_avg_inference_time_ms"]) for r in paired]

    x = list(range(len(model_names)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    mobile_bars = ax.bar([i - width / 2 for i in x], mobile_times, width=width, label="Mobile")
    pc_bars = ax.bar([i + width / 2 for i in x], pc_times, width=width, label="PC")

    gt = (gt_kind or "coco").strip().lower()
    gt_desc = "GT COCO (annotations JSON)" if gt == "coco" else "GT YOLO (imagens + labels .txt)"
    ax.set_title(f"Tempo médio de inferência: mobile vs PC\n({gt_desc}; mobile = export app, PC = inferência local)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Tempo médio de inferência (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    for bars in (mobile_bars, pc_bars):
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    if save:
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Gráfico Mobile vs PC salvo em: {out_path}")
    plt.close(fig)


def plot_mobile_vs_pc_fps_bars(
    rows: List[Dict[str, Any]],
    save: bool,
    output_dir: Path,
    filename: str = "comparison_fps_mean_mobile_vs_pc.png",
    gt_kind: str = "coco",
) -> None:
    paired = [
        r
        for r in rows
        if _fps_from_ms(r.get("avg_inference_time_ms")) is not None
        and _fps_from_ms(r.get("pc_avg_inference_time_ms")) is not None
    ]
    if not paired:
        print("Nenhum par mobile/PC com FPS médio disponível para plotar.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"] for r in paired]
    mobile_fps = [float(_fps_from_ms(r.get("avg_inference_time_ms")) or 0.0) for r in paired]
    pc_fps = [float(_fps_from_ms(r.get("pc_avg_inference_time_ms")) or 0.0) for r in paired]

    x = list(range(len(model_names)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    mobile_bars = ax.bar([i - width / 2 for i in x], mobile_fps, width=width, label="Mobile")
    pc_bars = ax.bar([i + width / 2 for i in x], pc_fps, width=width, label="PC")

    gt = (gt_kind or "coco").strip().lower()
    gt_desc = "GT COCO (annotations JSON)" if gt == "coco" else "GT YOLO (imagens + labels .txt)"
    ax.set_title(f"FPS médio: mobile vs PC\n({gt_desc}; derivado de 1000 / tempo médio)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("FPS médio")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    for bars in (mobile_bars, pc_bars):
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    if save:
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Gráfico FPS médio Mobile vs PC salvo em: {out_path}")
    plt.close(fig)


def plot_mobile_vs_pc_map50_bars(
    rows: List[Dict[str, Any]],
    save: bool,
    output_dir: Path,
    filename: str = "comparison_map50_mobile_vs_pc.png",
    gt_kind: str = "coco",
) -> None:
    paired = [
        r
        for r in rows
        if r.get("mAP50") is not None and r.get("pc_mAP50") is not None
    ]
    if not paired:
        print("Nenhum par mobile/PC com mAP50 disponível para plotar.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"] for r in paired]
    mobile_map = [float(r["mAP50"]) for r in paired]
    pc_map = [float(r["pc_mAP50"]) for r in paired]

    x = list(range(len(model_names)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    mobile_bars = ax.bar([i - width / 2 for i in x], mobile_map, width=width, label="Mobile (mAP50 vs GT)")
    pc_bars = ax.bar([i + width / 2 for i in x], pc_map, width=width, label="PC (mAP50 vs GT)")

    gt = (gt_kind or "coco").strip().lower()
    gt_desc = "GT COCO (annotations JSON)" if gt == "coco" else "GT YOLO (imagens + labels .txt)"
    ax.set_title(
        f"mAP50 face ao mesmo ground truth: mobile vs PC\n"
        f"({gt_desc}; mobile = predições do JSON da app, PC = predições locais + COCOeval)"
    )
    ax.set_xlabel("Modelo")
    ax.set_ylabel("mAP50")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    for bars in (mobile_bars, pc_bars):
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    if save:
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Gráfico mAP50 Mobile vs PC salvo em: {out_path}")
    plt.close(fig)


def _model_candidates_from_row(model_name: str, pred_file: str) -> List[str]:
    stem = Path(pred_file).stem
    candidates = [
        str(model_name),
        stem,
        f"{stem}.tflite",
        re.sub(r"^(coco|epi)[-_]?val[-_]?", "", stem, flags=re.IGNORECASE),
        re.sub(r"^(coco|epi)[-_]?val[-_]?", "", stem, flags=re.IGNORECASE) + ".tflite",
        stem.replace("COCO_val_", "").replace("EPI_val_", ""),
        stem.replace("COCO_val_", "").replace("EPI_val_", "") + ".tflite",
    ]
    cleaned: List[str] = []
    seen = set()
    for cand in candidates:
        c = str(cand).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        cleaned.append(c)
    return cleaned


def _resolve_model_path_for_row(cfg: Dict[str, Any], model_name: str, pred_file: str) -> Path:
    errors: List[str] = []
    for candidate in _model_candidates_from_row(model_name, pred_file):
        try:
            return resolve_model_path_string(cfg, candidate)
        except Exception as e:
            errors.append(f"{candidate}: {e}")
    raise FileNotFoundError(
        "Não foi possível resolver o modelo para este JSON de predição.\n"
        f"  model={model_name!r}\n"
        f"  pred_file={pred_file!r}\n"
        "Tentativas:\n  - " + "\n  - ".join(errors)
    )


def _run_pc_eval_for_model(
    cfg: Dict[str, Any],
    gt_kind: str,
    model_path: Path,
    model_name: str,
    output_dir: Path,
    gt_coco: Optional[str],
    yolo_imgs: Optional[str],
    yolo_lbls: Optional[str],
    yolo_class_names: Any,
) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    inf = cfg.get("inference", {})
    plots_cfg = cfg.get("plots", {})
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_") or "model"
    imgsz = int(inf.get("imgsz", 640))
    conf = float(plots_cfg.get("pc_comparison_conf", inf.get("conf", 0.001)))
    iou_nms = float(inf.get("iou_nms", 0.7))
    device = inf.get("device")
    max_det = plots_cfg.get("pc_comparison_max_det", 100)
    max_images = plots_cfg.get("pc_comparison_max_images")
    max_det = int(max_det) if max_det is not None else None
    max_images = int(max_images) if max_images is not None else None

    if gt_kind == "yolo":
        if not yolo_imgs or not yolo_lbls:
            raise ValueError("Para gt_kind=yolo, defina paths.mobile_gt_yolo_images e paths.mobile_gt_yolo_labels.")
        return evaluate_yolo_with_yolo_gt(
            model_path=str(model_path),
            images_dir=yolo_imgs,
            labels_dir=yolo_lbls,
            output_dir=output_dir / "pc_eval" / safe_model,
            imgsz=imgsz,
            conf_thres=conf,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
            max_images=max_images,
            gt_class_names=yolo_class_names,
            class_id_to_gt_class_id=inf.get("class_id_to_gt_class_id"),
            class_name_to_gt_class_id=inf.get("class_name_to_gt_class_id"),
        )

    if not gt_coco:
        raise ValueError("Para gt_kind=coco, defina paths.mobile_gt_coco_json (ou plots.gt_json_for_mobile_plots).")
    val_images_dir = paths.get("val_images_dir")
    if not val_images_dir:
        raise ValueError("Para inferência no PC com gt_kind=coco, defina paths.val_images_dir.")
    return inference_and_evaluate_coco_gt(
        model_path=str(model_path),
        val_images_dir=val_images_dir,
        ann_file=gt_coco,
        out_predictions_json=output_dir / "pc_predictions" / f"{safe_model}.json",
        imgsz=imgsz,
        conf_thres=conf,
        iou_nms=iou_nms,
        device=device,
        max_det=max_det,
        max_images=max_images,
        class_id_to_category_id=inf.get("class_id_to_category_id"),
        class_name_to_category_id=inf.get("class_name_to_category_id"),
        class_mapping_mode=str(inf.get("class_mapping_mode", "auto")).strip().lower(),
    )


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
    include_pc = bool(cfg.get("plots", {}).get("include_pc_comparison", False))

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
        timing = eval_result.get("timing", {})
        comparison_rows.append(
            {
                "file": pred_file,
                "model": model_name,
                "mAP50": map50,
                "avg_inference_time_ms": timing.get("avg_inference_time_ms"),
                "median_inference_time_ms": timing.get("median_inference_time_ms"),
                "p95_inference_time_ms": timing.get("p95_inference_time_ms"),
                "p99_inference_time_ms": timing.get("p99_inference_time_ms"),
                "std_inference_time_ms": timing.get("std_inference_time_ms"),
                "pc_mAP50": None,
                "pc_avg_inference_time_ms": None,
                "pc_median_inference_time_ms": None,
                "pc_p95_inference_time_ms": None,
                "pc_p99_inference_time_ms": None,
                "pc_std_inference_time_ms": None,
                "full_result": eval_result,
            }
        )

    if include_pc:
        print("\n=== Inferência no PC para comparação Mobile vs PC ===")
        for row in comparison_rows:
            model_name = str(row["model"])
            pred_file = str(row["file"])
            print(f"Modelo PC: {model_name} (origem: {pred_file})")
            model_path = _resolve_model_path_for_row(cfg, model_name, pred_file)
            pc_result = _run_pc_eval_for_model(
                cfg=cfg,
                gt_kind=gt_kind,
                model_path=model_path,
                model_name=model_name,
                output_dir=out_dir,
                gt_coco=gt_coco,
                yolo_imgs=yolo_imgs,
                yolo_lbls=yolo_lbls,
                yolo_class_names=yolo_class_names,
            )
            pc_metrics = pc_result.get("metrics", {})
            pc_timing = pc_result.get("timing", {})
            row["pc_mAP50"] = pc_metrics.get("mAP_50")
            row["pc_avg_inference_time_ms"] = pc_timing.get("avg_inference_time_ms")
            row["pc_median_inference_time_ms"] = pc_timing.get("median_inference_time_ms")
            row["pc_p95_inference_time_ms"] = pc_timing.get("p95_inference_time_ms")
            row["pc_p99_inference_time_ms"] = pc_timing.get("p99_inference_time_ms")
            row["pc_std_inference_time_ms"] = pc_timing.get("std_inference_time_ms")
            row["pc_full_result"] = pc_result

    print("\n=== Resumo ===")
    for row in comparison_rows:
        t = row["avg_inference_time_ms"]
        ts = f"{t:.3f}" if t is not None else "n/a"
        pc_t = row.get("pc_avg_inference_time_ms")
        pc_ts = f"{pc_t:.3f}" if pc_t is not None else "n/a"
        pc_map = row.get("pc_mAP50")
        pc_map_s = f"{float(pc_map):.6f}" if pc_map is not None else "n/a"
        print(
            f"Modelo: {row['model']}\n"
            f"  mAP50 (mobile): {row['mAP50']:.6f}\n"
            f"  Avg inference time mobile (ms): {ts}\n"
            f"  mAP50 (pc): {pc_map_s}\n"
            f"  Avg inference time pc (ms): {pc_ts}\n"
        )

    plot_comparison_bars(comparison_rows, save=save, output_dir=out_dir)
    plot_fps_bars(comparison_rows, save=save, output_dir=out_dir)
    if include_pc:
        plot_mobile_vs_pc_map50_bars(comparison_rows, save=save, output_dir=out_dir, gt_kind=str(gt_kind))
        plot_mobile_vs_pc_inference_bars(comparison_rows, save=save, output_dir=out_dir, gt_kind=str(gt_kind))
        plot_mobile_vs_pc_fps_bars(comparison_rows, save=save, output_dir=out_dir, gt_kind=str(gt_kind))

    summary_output = [
        {
            "file": r["file"],
            "model": r["model"],
            "mAP50": r["mAP50"],
            "avg_inference_time_ms": r["avg_inference_time_ms"],
            "mAP50_mobile": r["mAP50"],
            "avg_inference_time_ms_mobile": r["avg_inference_time_ms"],
            "median_inference_time_ms_mobile": r.get("median_inference_time_ms"),
            "p95_inference_time_ms_mobile": r.get("p95_inference_time_ms"),
            "p99_inference_time_ms_mobile": r.get("p99_inference_time_ms"),
            "std_inference_time_ms_mobile": r.get("std_inference_time_ms"),
            "mAP50_pc": r.get("pc_mAP50"),
            "avg_inference_time_ms_pc": r.get("pc_avg_inference_time_ms"),
            "median_inference_time_ms_pc": r.get("pc_median_inference_time_ms"),
            "p95_inference_time_ms_pc": r.get("pc_p95_inference_time_ms"),
            "p99_inference_time_ms_pc": r.get("pc_p99_inference_time_ms"),
            "std_inference_time_ms_pc": r.get("pc_std_inference_time_ms"),
            "fps_mean_mobile": _fps_from_ms(r.get("avg_inference_time_ms")),
            "fps_mean_pc": _fps_from_ms(r.get("pc_avg_inference_time_ms")),
        }
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
