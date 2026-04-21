"""Avaliação de predições exportadas pelo app (JSON) contra GT COCO ou GT YOLO."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_mobile_predictions(
    gt_json_path: str | Path,
    pred_json_path: str | Path,
    class_name_aliases: Optional[Dict[str, str]] = None,
    class_name_to_category_id: Optional[Dict[str, int]] = None,
    iou_thr_aux: float = 0.5,
    save_converted_predictions_path: Optional[str | Path] = None,
    save_filtered_gt_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    class_name_aliases = class_name_aliases or {}

    def normalize_name(name: str) -> str:
        name = str(name).strip().lower().replace("_", " ")
        return " ".join(name.split())

    def apply_alias(name: str, aliases: dict) -> str:
        norm = normalize_name(name)
        normalized_aliases = {normalize_name(k): normalize_name(v) for k, v in aliases.items()}
        return normalized_aliases.get(norm, norm)

    def find_image_key(item: dict):
        for key in ["file", "image", "image_path", "imagePath", "filename", "file_name", "path"]:
            if key in item:
                return key
        return None

    def find_boxes_key(item: dict):
        for key in ["boxes", "detections", "predictions", "results"]:
            if key in item and isinstance(item[key], list):
                return key
        return None

    def extract_box_fields(box: dict):
        cls_name = (
            box.get("className")
            or box.get("class_name")
            or box.get("class")
            or box.get("label")
            or box.get("category")
            or ""
        )
        score = (
            box.get("confidence")
            if box.get("confidence") is not None
            else box.get("score")
            if box.get("score") is not None
            else box.get("conf")
            if box.get("conf") is not None
            else 0.0
        )
        if all(k in box for k in ["x1", "y1", "x2", "y2"]):
            return cls_name, float(score), float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        if all(k in box for k in ["xmin", "ymin", "xmax", "ymax"]):
            return (
                cls_name,
                float(score),
                float(box["xmin"]),
                float(box["ymin"]),
                float(box["xmax"]),
                float(box["ymax"]),
            )
        if all(k in box for k in ["x", "y", "w", "h"]):
            x1 = float(box["x"])
            y1 = float(box["y"])
            x2 = x1 + float(box["w"])
            y2 = y1 + float(box["h"])
            return cls_name, float(score), x1, y1, x2, y2
        if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) == 4:
            b = [float(v) for v in box["bbox"]]
            x1, y1, x2, y2 = b
            if x2 < x1 or y2 < y1:
                x1, y1, w, h = b
                x2 = x1 + w
                y2 = y1 + h
            return cls_name, float(score), x1, y1, x2, y2
        raise ValueError(f"Formato de bbox não reconhecido: {box}")

    def load_prediction_items(pred_data):
        if isinstance(pred_data, list):
            return pred_data
        for key in ["results", "predictions", "items"]:
            if key in pred_data and isinstance(pred_data[key], list):
                return pred_data[key]
        raise ValueError("Não foi encontrada uma lista de predições no JSON.")

    def build_dataset_maps(coco_full):
        gt_images_by_name = {img["file_name"]: img for img in coco_full["images"]}
        name_to_cat_id = {normalize_name(cat["name"]): int(cat["id"]) for cat in coco_full["categories"]}
        cat_id_to_name = {int(cat["id"]): cat["name"] for cat in coco_full["categories"]}
        return gt_images_by_name, name_to_cat_id, cat_id_to_name

    def resolve_category_id(cls_name_raw, name_to_cat_id, aliases=None, manual_mapping=None):
        aliases = aliases or {}
        cls_name_norm = apply_alias(cls_name_raw, aliases)
        if manual_mapping is not None:
            normalized_manual = {normalize_name(k): int(v) for k, v in manual_mapping.items()}
            if cls_name_norm in normalized_manual:
                return normalized_manual[cls_name_norm]
        if cls_name_norm in name_to_cat_id:
            return name_to_cat_id[cls_name_norm]
        return None

    def xywh_to_xyxy(box):
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    with open(gt_json_path, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    pred_items = load_prediction_items(pred_data)
    gt_images_by_name, name_to_cat_id, cat_id_to_name = build_dataset_maps(coco_full)

    pred_filenames = set()
    items_without_image_field = 0
    for item in pred_items:
        image_key = find_image_key(item)
        if image_key is None:
            items_without_image_field += 1
            continue
        fname = Path(item[image_key]).name
        if fname in gt_images_by_name:
            pred_filenames.add(fname)

    filtered_images = [img for img in coco_full["images"] if img["file_name"] in pred_filenames]
    filtered_image_ids = {img["id"] for img in filtered_images}
    filtered_annotations = [ann for ann in coco_full["annotations"] if ann["image_id"] in filtered_image_ids]
    filtered_coco = {
        "info": coco_full.get("info", {}),
        "licenses": coco_full.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_full["categories"],
    }
    if save_filtered_gt_path is not None:
        with open(save_filtered_gt_path, "w", encoding="utf-8") as f:
            json.dump(filtered_coco, f)

    converted_predictions: List[dict] = []
    missing_images = []
    unknown_classes = defaultdict(int)
    items_without_boxes = 0
    invalid_boxes = 0

    for item in pred_items:
        image_key = find_image_key(item)
        if image_key is None:
            continue
        file_name = Path(item[image_key]).name
        if file_name not in gt_images_by_name:
            missing_images.append(file_name)
            continue
        image_id = gt_images_by_name[file_name]["id"]
        boxes_key = find_boxes_key(item)
        if boxes_key is None:
            items_without_boxes += 1
            continue
        for box in item.get(boxes_key, []):
            try:
                cls_name_raw, score, x1, y1, x2, y2 = extract_box_fields(box)
            except Exception:
                invalid_boxes += 1
                continue
            category_id = resolve_category_id(
                cls_name_raw=cls_name_raw,
                name_to_cat_id=name_to_cat_id,
                aliases=class_name_aliases,
                manual_mapping=class_name_to_category_id,
            )
            if category_id is None:
                unknown_classes[str(cls_name_raw)] += 1
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                invalid_boxes += 1
                continue
            converted_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    if save_converted_predictions_path is not None:
        with open(save_converted_predictions_path, "w", encoding="utf-8") as f:
            json.dump(converted_predictions, f)

    if len(converted_predictions) == 0:
        raise ValueError(
            "Nenhuma predição válida foi convertida. Verifique nomes das classes, imagens e formato do JSON."
        )

    tmp_dir: Optional[str] = None
    try:
        if save_filtered_gt_path is None or save_converted_predictions_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="bench_mobile_coco_")
        temp_gt_path = Path(save_filtered_gt_path) if save_filtered_gt_path else Path(tmp_dir) / "filtered_gt.json"
        temp_pred_path = (
            Path(save_converted_predictions_path)
            if save_converted_predictions_path
            else Path(tmp_dir) / "converted_preds.json"
        )
        if save_filtered_gt_path is None:
            with open(temp_gt_path, "w", encoding="utf-8") as f:
                json.dump(filtered_coco, f)
        if save_converted_predictions_path is None:
            with open(temp_pred_path, "w", encoding="utf-8") as f:
                json.dump(converted_predictions, f)

        coco_gt = COCO(str(temp_gt_path))
        coco_dt = coco_gt.loadRes(str(temp_pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    coco_metrics = {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }

    gt_by_img_cat = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        gt_by_img_cat[(ann["image_id"], ann["category_id"])].append(ann)

    pred_by_img_cat = defaultdict(list)
    for pred in converted_predictions:
        pred_by_img_cat[(pred["image_id"], pred["category_id"])].append(pred)
    for k in pred_by_img_cat:
        pred_by_img_cat[k] = sorted(pred_by_img_cat[k], key=lambda x: x["score"], reverse=True)

    TP, FP, FN = 0, 0, 0
    all_keys = set(gt_by_img_cat.keys()) | set(pred_by_img_cat.keys())
    for key in all_keys:
        gts = gt_by_img_cat.get(key, [])
        preds = pred_by_img_cat.get(key, [])
        matched_gt = [False] * len(gts)
        for pred in preds:
            pred_box = xywh_to_xyxy(pred["bbox"])
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gts):
                if matched_gt[i]:
                    continue
                gt_box = xywh_to_xyxy(gt["bbox"])
                iou = iou_xyxy(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thr_aux and best_idx >= 0:
                TP += 1
                matched_gt[best_idx] = True
            else:
                FP += 1
        FN += matched_gt.count(False)

    precision_aux = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_aux = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy_aux = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1_aux = (
        2 * precision_aux * recall_aux / (precision_aux + recall_aux)
        if (precision_aux + recall_aux) > 0
        else 0.0
    )
    aux_metrics = {
        "accuracy_aux": accuracy_aux,
        "precision_aux": precision_aux,
        "recall_aux": recall_aux,
        "f1_aux": f1_aux,
        "iou_threshold": iou_thr_aux,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }

    inference_times = []
    for item in pred_items:
        t = item.get("inference_time_ms")
        if t is not None:
            inference_times.append(float(t))
    timing = {
        "num_images_with_time": len(inference_times),
        "avg_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else None,
        "min_inference_time_ms": min(inference_times) if inference_times else None,
        "max_inference_time_ms": max(inference_times) if inference_times else None,
    }

    return {
        "model": pred_data.get("model") if isinstance(pred_data, dict) else None,
        "num_prediction_items": len(pred_items),
        "num_images_evaluated": len(filtered_images),
        "num_converted_predictions": len(converted_predictions),
        "coco_metrics": coco_metrics,
        "aux_metrics": aux_metrics,
        "timing": timing,
        "unknown_classes": dict(unknown_classes),
        "missing_images_count": len(missing_images),
        "missing_images_examples": missing_images[:20],
        "items_without_image_field": items_without_image_field,
        "items_without_boxes": items_without_boxes,
        "invalid_boxes": invalid_boxes,
    }


def evaluate_mobile_predictions_yolo(
    gt_images_dir: str | Path,
    gt_labels_dir: str | Path,
    pred_json_path: str | Path,
    class_names: Optional[List[str] | Dict[int, str]] = None,
    dataset_yaml_path: Optional[str | Path] = None,
    class_name_aliases: Optional[Dict[str, str]] = None,
    class_name_to_category_id: Optional[Dict[str, int]] = None,
    iou_thr_aux: float = 0.5,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    save_converted_gt_coco_path: Optional[str | Path] = None,
    save_converted_predictions_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    class_name_aliases = class_name_aliases or {}

    def normalize_name(name: str) -> str:
        name = str(name).strip().lower().replace("_", " ")
        return " ".join(name.split())

    def apply_alias(name: str, aliases: dict) -> str:
        norm = normalize_name(name)
        normalized_aliases = {normalize_name(k): normalize_name(v) for k, v in aliases.items()}
        return normalized_aliases.get(norm, norm)

    def load_class_names(class_names=None, dataset_yaml_path=None):
        if class_names is not None:
            if isinstance(class_names, dict):
                return {int(k): str(v) for k, v in class_names.items()}
            if isinstance(class_names, list):
                return {i: str(v) for i, v in enumerate(class_names)}
            raise ValueError("class_names deve ser list[str] ou dict[int, str].")
        if dataset_yaml_path is None:
            raise ValueError("Forneça class_names ou dataset_yaml_path.")
        with open(dataset_yaml_path, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)
        names = data_yaml.get("names")
        if names is None:
            raise ValueError("data.yaml não possui a chave 'names'.")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        raise ValueError("Formato inválido de 'names' no data.yaml.")

    def find_image_for_stem(images_dir: Path, stem: str):
        for ext in image_extensions:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def yolo_to_xywh_abs(xc, yc, w, h, img_w, img_h):
        bw = w * img_w
        bh = h * img_h
        x1 = (xc * img_w) - bw / 2
        y1 = (yc * img_h) - bh / 2
        return [float(x1), float(y1), float(bw), float(bh)]

    def xywh_to_xyxy(box):
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def find_image_key(item: dict):
        for key in ["file", "image", "image_path", "imagePath", "filename", "file_name", "path"]:
            if key in item:
                return key
        return None

    def find_boxes_key(item: dict):
        for key in ["boxes", "detections", "predictions", "results"]:
            if key in item and isinstance(item[key], list):
                return key
        return None

    def extract_box_fields(box: dict):
        cls_name = (
            box.get("className")
            or box.get("class_name")
            or box.get("class")
            or box.get("label")
            or box.get("category")
            or ""
        )
        score = (
            box.get("confidence")
            if box.get("confidence") is not None
            else box.get("score")
            if box.get("score") is not None
            else box.get("conf")
            if box.get("conf") is not None
            else 0.0
        )
        if all(k in box for k in ["x1", "y1", "x2", "y2"]):
            return cls_name, float(score), float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        if all(k in box for k in ["xmin", "ymin", "xmax", "ymax"]):
            return (
                cls_name,
                float(score),
                float(box["xmin"]),
                float(box["ymin"]),
                float(box["xmax"]),
                float(box["ymax"]),
            )
        if all(k in box for k in ["x", "y", "w", "h"]):
            x1 = float(box["x"])
            y1 = float(box["y"])
            x2 = x1 + float(box["w"])
            y2 = y1 + float(box["h"])
            return cls_name, float(score), x1, y1, x2, y2
        if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) == 4:
            b = [float(v) for v in box["bbox"]]
            x1, y1, x2, y2 = b
            if x2 < x1 or y2 < y1:
                x1, y1, w, h = b
                x2 = x1 + w
                y2 = y1 + h
            return cls_name, float(score), x1, y1, x2, y2
        raise ValueError(f"Formato de bbox não reconhecido: {box}")

    def load_prediction_items(pred_data):
        if isinstance(pred_data, list):
            return pred_data
        for key in ["results", "predictions", "items"]:
            if key in pred_data and isinstance(pred_data[key], list):
                return pred_data[key]
        raise ValueError("Não foi encontrada uma lista de predições no JSON.")

    gt_images_dir = Path(gt_images_dir)
    gt_labels_dir = Path(gt_labels_dir)
    model_classes = load_class_names(class_names=class_names, dataset_yaml_path=dataset_yaml_path)
    model_name_to_catid = {normalize_name(name): idx + 1 for idx, name in model_classes.items()}

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    filename_to_image_id = {}
    label_files = sorted(gt_labels_dir.glob("*.txt"))
    for image_id, label_file in enumerate(label_files, start=1):
        stem = label_file.stem
        image_path = find_image_for_stem(gt_images_dir, stem)
        if image_path is None:
            continue
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": img_w, "height": img_h})
        filename_to_image_id[image_path.name] = image_id
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            if cls_id not in model_classes:
                continue
            bbox = yolo_to_xywh_abs(xc, yc, w, h, img_w, img_h)
            bw, bh = bbox[2], bbox[3]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id + 1,
                    "bbox": bbox,
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco_categories = [{"id": idx + 1, "name": name} for idx, name in model_classes.items()]
    coco_gt_dict = {
        "info": {},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    if save_converted_gt_coco_path is not None:
        with open(save_converted_gt_coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_gt_dict, f)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    pred_items = load_prediction_items(pred_data)

    converted_predictions: List[dict] = []
    missing_images = []
    unknown_classes = defaultdict(int)
    items_without_image_field = 0
    items_without_boxes = 0
    invalid_boxes = 0

    def resolve_category_id(cls_name_raw):
        cls_name_norm = apply_alias(cls_name_raw, class_name_aliases)
        if class_name_to_category_id is not None:
            normalized_manual = {normalize_name(k): int(v) for k, v in class_name_to_category_id.items()}
            if cls_name_norm in normalized_manual:
                return normalized_manual[cls_name_norm]
        return model_name_to_catid.get(cls_name_norm)

    for item in pred_items:
        image_key = find_image_key(item)
        if image_key is None:
            items_without_image_field += 1
            continue
        file_name = Path(item[image_key]).name
        if file_name not in filename_to_image_id:
            missing_images.append(file_name)
            continue
        image_id = filename_to_image_id[file_name]
        boxes_key = find_boxes_key(item)
        if boxes_key is None:
            items_without_boxes += 1
            continue
        for box in item.get(boxes_key, []):
            try:
                cls_name_raw, score, x1, y1, x2, y2 = extract_box_fields(box)
            except Exception:
                invalid_boxes += 1
                continue
            category_id = resolve_category_id(cls_name_raw)
            if category_id is None:
                unknown_classes[str(cls_name_raw)] += 1
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                invalid_boxes += 1
                continue
            converted_predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    if len(converted_predictions) == 0:
        raise ValueError("Nenhuma predição válida foi convertida.")

    if save_converted_predictions_path is not None:
        with open(save_converted_predictions_path, "w", encoding="utf-8") as f:
            json.dump(converted_predictions, f)

    tmp_dir: Optional[str] = None
    try:
        if save_converted_gt_coco_path is None or save_converted_predictions_path is None:
            tmp_dir = tempfile.mkdtemp(prefix="bench_mobile_yolo_")
        temp_gt_path = Path(save_converted_gt_coco_path) if save_converted_gt_coco_path else Path(tmp_dir) / "gt_coco.json"
        temp_pred_path = (
            Path(save_converted_predictions_path)
            if save_converted_predictions_path
            else Path(tmp_dir) / "preds.json"
        )
        if save_converted_gt_coco_path is None:
            with open(temp_gt_path, "w", encoding="utf-8") as f:
                json.dump(coco_gt_dict, f)
        if save_converted_predictions_path is None:
            with open(temp_pred_path, "w", encoding="utf-8") as f:
                json.dump(converted_predictions, f)

        coco_gt = COCO(str(temp_gt_path))
        coco_dt = coco_gt.loadRes(str(temp_pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    coco_metrics = {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }

    gt_by_img_cat = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        gt_by_img_cat[(ann["image_id"], ann["category_id"])].append(ann)
    pred_by_img_cat = defaultdict(list)
    for pred in converted_predictions:
        pred_by_img_cat[(pred["image_id"], pred["category_id"])].append(pred)
    for k in pred_by_img_cat:
        pred_by_img_cat[k] = sorted(pred_by_img_cat[k], key=lambda x: x["score"], reverse=True)

    TP, FP, FN = 0, 0, 0
    all_keys = set(gt_by_img_cat.keys()) | set(pred_by_img_cat.keys())
    for key in all_keys:
        gts = gt_by_img_cat.get(key, [])
        preds = pred_by_img_cat.get(key, [])
        matched_gt = [False] * len(gts)
        for pred in preds:
            pred_box = xywh_to_xyxy(pred["bbox"])
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gts):
                if matched_gt[i]:
                    continue
                gt_box = xywh_to_xyxy(gt["bbox"])
                iou = iou_xyxy(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thr_aux and best_idx >= 0:
                TP += 1
                matched_gt[best_idx] = True
            else:
                FP += 1
        FN += matched_gt.count(False)

    precision_aux = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_aux = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy_aux = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1_aux = (
        2 * precision_aux * recall_aux / (precision_aux + recall_aux)
        if (precision_aux + recall_aux) > 0
        else 0.0
    )
    aux_metrics = {
        "accuracy_aux": accuracy_aux,
        "precision_aux": precision_aux,
        "recall_aux": recall_aux,
        "f1_aux": f1_aux,
        "iou_threshold": iou_thr_aux,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }

    inference_times = []
    for item in pred_items:
        t = item.get("inference_time_ms")
        if t is not None:
            inference_times.append(float(t))
    timing = {
        "num_images_with_time": len(inference_times),
        "avg_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else None,
        "min_inference_time_ms": min(inference_times) if inference_times else None,
        "max_inference_time_ms": max(inference_times) if inference_times else None,
    }

    return {
        "model": pred_data.get("model") if isinstance(pred_data, dict) else None,
        "num_prediction_items": len(pred_items),
        "num_gt_images": len(coco_images),
        "num_gt_annotations": len(coco_annotations),
        "num_converted_predictions": len(converted_predictions),
        "coco_metrics": coco_metrics,
        "aux_metrics": aux_metrics,
        "timing": timing,
        "unknown_classes": dict(unknown_classes),
        "missing_images_count": len(missing_images),
        "missing_images_examples": missing_images[:20],
        "items_without_image_field": items_without_image_field,
        "items_without_boxes": items_without_boxes,
        "invalid_boxes": invalid_boxes,
    }
