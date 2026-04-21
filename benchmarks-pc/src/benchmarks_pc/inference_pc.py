"""Inferência YOLO no PC + COCOeval (GT em JSON COCO ou labels YOLO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Mapeamento classe YOLO 0..79 -> category_id COCO (dataset COCO val2017)
COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def ensure_pc_backend_for_weights(model_path: Path) -> None:
    """
    Ultralytics carrega .tflite via TensorFlow/tflite_runtime.
    .pt/.onnx usam PyTorch/ONNXRuntime (já cobertos pelas dependências base ou wheels).
    """
    suf = model_path.suffix.lower()
    if suf != ".tflite":
        return
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]

        del Interpreter
        return
    except ImportError:
        pass
    try:
        import tensorflow as tf  # noqa: F401

        _ = tf
        return
    except ImportError:
        raise SystemExit(
            "Inferência no PC com ficheiro .tflite requer TensorFlow (ou tflite_runtime).\n"
            "  poetry install -E tflite\n"
            "Alternativa: em paths.model_for_infer_coco / model_for_infer_epi use um modelo .pt "
            "(recomendado no PC; reserve .tflite para a app Android)."
        ) from None


def evaluate_coco(gt_json_path: str | Path, pred_json_path: str | Path) -> Tuple[Any, Dict[str, float]]:
    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
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
    return coco_eval, metrics


def build_filename_to_image_id(coco_gt: COCO) -> Dict[str, int]:
    imgs = coco_gt.loadImgs(coco_gt.getImgIds())
    return {img["file_name"]: img["id"] for img in imgs}


def build_dataset_category_maps(coco_gt: COCO) -> Tuple[Dict[str, int], Dict[int, str]]:
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    dataset_name_to_catid: Dict[str, int] = {}
    dataset_catid_to_name: Dict[int, str] = {}
    for cat in cats:
        cat_id = int(cat["id"])
        cat_name = cat["name"]
        dataset_catid_to_name[cat_id] = cat_name
        dataset_name_to_catid[normalize_name(cat_name)] = cat_id
    return dataset_name_to_catid, dataset_catid_to_name


def build_model_class_maps(model: YOLO) -> Tuple[Dict[int, str], Dict[str, int]]:
    names = model.names
    if isinstance(names, dict):
        model_id_to_name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        model_id_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"Formato inesperado de model.names: {type(names)}")
    model_name_to_id = {normalize_name(v): k for k, v in model_id_to_name.items()}
    return model_id_to_name, model_name_to_id


def resolve_class_mapping(
    model: YOLO,
    coco_gt: COCO,
    class_id_to_category_id: Optional[Dict[int, int]] = None,
    class_name_to_category_id: Optional[Dict[str, int]] = None,
) -> Dict[int, int]:
    model_id_to_name, _ = build_model_class_maps(model)
    dataset_name_to_catid, dataset_catid_to_name = build_dataset_category_maps(coco_gt)

    if class_id_to_category_id is not None:
        mapping: Dict[int, int] = {}
        for cls_id, cat_id in class_id_to_category_id.items():
            cls_id = int(cls_id)
            cat_id = int(cat_id)
            if cls_id not in model_id_to_name:
                raise ValueError(f"Classe do modelo {cls_id} não existe em model.names")
            if cat_id not in dataset_catid_to_name:
                raise ValueError(f"category_id {cat_id} não existe no annotation JSON")
            mapping[cls_id] = cat_id
        return mapping

    if class_name_to_category_id is not None:
        mapping = {}
        normalized_manual = {normalize_name(k): int(v) for k, v in class_name_to_category_id.items()}
        for cls_id, cls_name in model_id_to_name.items():
            norm_name = normalize_name(cls_name)
            if norm_name in normalized_manual:
                mapping[cls_id] = normalized_manual[norm_name]
        if not mapping:
            raise ValueError("Nenhuma classe do modelo bateu com class_name_to_category_id.")
        return mapping

    mapping = {}
    unmatched: List[Tuple[int, str]] = []
    for cls_id, cls_name in model_id_to_name.items():
        norm_name = normalize_name(cls_name)
        if norm_name in dataset_name_to_catid:
            mapping[cls_id] = dataset_name_to_catid[norm_name]
        else:
            unmatched.append((cls_id, cls_name))

    print("=== Resolução automática de classes ===")
    print(f"Classes do modelo: {len(model_id_to_name)}")
    print(f"Categorias no dataset: {len(dataset_catid_to_name)}")
    print(f"Classes mapeadas automaticamente: {len(mapping)}")
    if unmatched:
        print("\nClasses do modelo sem correspondência no dataset:")
        for cls_id, cls_name in unmatched[:20]:
            print(f"  - {cls_id}: {cls_name}")
        if len(unmatched) > 20:
            print(f"  ... e mais {len(unmatched) - 20}")
    if not mapping:
        raise ValueError(
            "Não foi possível mapear nenhuma classe automaticamente.\n"
            "Forneça class_id_to_category_id ou class_name_to_category_id."
        )
    return mapping


def collect_image_paths(images_dir: str | Path) -> List[Path]:
    images_dir = Path(images_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])


def run_yolo_predictions_to_coco_json(
    model: YOLO,
    image_paths: List[Path],
    coco_gt: COCO,
    class_mapping: Dict[int, int],
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
) -> List[dict]:
    filename_to_image_id = build_filename_to_image_id(coco_gt)
    predictions: List[dict] = []
    skipped_no_gt = 0
    skipped_unmapped_class = 0
    for img_path in tqdm(image_paths, desc="Inferência"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            skipped_no_gt += 1
            continue
        image_id = filename_to_image_id[file_name]
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_nms,
            device=device,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, clss):
            if cls_id not in class_mapping:
                skipped_unmapped_class += 1
                continue
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(class_mapping[int(cls_id)]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
    print(f"\nImagens sem GT correspondente pelo filename: {skipped_no_gt}")
    print(f"Detecções ignoradas por classe não mapeada: {skipped_unmapped_class}")
    print(f"Total de detecções válidas: {len(predictions)}")
    return predictions


def run_yolo_predictions_coco80_mapping(
    model: YOLO,
    image_paths: List[Path],
    coco_gt: COCO,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
) -> List[dict]:
    """Para modelos treinados no conjunto COCO com 80 classes na ordem padrão YOLO."""
    filename_to_image_id = build_filename_to_image_id(coco_gt)
    predictions: List[dict] = []
    for img_path in tqdm(image_paths, desc="Inferência (COCO80)"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            continue
        image_id = filename_to_image_id[file_name]
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_nms,
            device=device,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, clss):
            cid = int(cls_id)
            if cid < 0 or cid >= len(COCO80_TO_COCO91):
                continue
            coco_cat_id = COCO80_TO_COCO91[cid]
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(coco_cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
    return predictions


def inference_and_evaluate_coco_gt(
    model_path: str | Path,
    val_images_dir: str | Path,
    ann_file: str | Path,
    out_predictions_json: str | Path,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
    class_id_to_category_id: Optional[Dict[int, int]] = None,
    class_name_to_category_id: Optional[Dict[str, int]] = None,
    class_mapping_mode: str = "auto",
) -> Dict[str, Any]:
    """
    class_mapping_mode: 'auto' usa nomes do modelo vs categorias do JSON COCO;
    'coco80' usa o mapeamento fixo 80 classes YOLO -> IDs COCO.
    """
    model_path = Path(model_path)
    ann_file = Path(ann_file)
    out_predictions_json = Path(out_predictions_json)
    out_predictions_json.parent.mkdir(parents=True, exist_ok=True)

    ensure_pc_backend_for_weights(model_path)
    print("Carregando modelo...")
    model = YOLO(str(model_path))
    print("Carregando ground truth COCO...")
    coco_gt = COCO(str(ann_file))

    if class_mapping_mode == "coco80":
        class_mapping = {i: COCO80_TO_COCO91[i] for i in range(len(COCO80_TO_COCO91))}
        print("Usando mapeamento COCO80 -> category_id COCO (modo coco80).")
    else:
        print("Resolvendo mapeamento de classes...")
        class_mapping = resolve_class_mapping(
            model=model,
            coco_gt=coco_gt,
            class_id_to_category_id=class_id_to_category_id,
            class_name_to_category_id=class_name_to_category_id,
        )

    print("\nMapeamento final model_class_id -> dataset_category_id:")
    names = model.names
    id_to_name = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    for k, v in sorted(class_mapping.items()):
        label = id_to_name.get(k, "?")
        print(f"  {k} ({label}) -> {v}")

    image_paths = collect_image_paths(val_images_dir)
    print(f"\nTotal de imagens encontradas: {len(image_paths)}")

    if class_mapping_mode == "coco80":
        predictions = run_yolo_predictions_coco80_mapping(
            model=model,
            image_paths=image_paths,
            coco_gt=coco_gt,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
        )
    else:
        predictions = run_yolo_predictions_to_coco_json(
            model=model,
            image_paths=image_paths,
            coco_gt=coco_gt,
            class_mapping=class_mapping,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
        )

    if len(predictions) == 0:
        raise ValueError(
            "Nenhuma predição válida foi gerada. Verifique modelo, thresholds, "
            "diretório de imagens ou mapeamento de classes."
        )

    with open(out_predictions_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"\nPredições salvas em: {out_predictions_json}")

    print("\nAvaliando com COCOeval...")
    _, metrics = evaluate_coco(ann_file, out_predictions_json)
    print("\n=== Métricas COCO ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("\n=== Resumo ===")
    print(f"mAP_50 (proxy precisão): {metrics['mAP_50']:.6f}")
    print(f"AR_100 (proxy recall):   {metrics['AR_100']:.6f}")
    return {"metrics": metrics, "predictions_path": str(out_predictions_json), "num_predictions": len(predictions)}


def evaluate_yolo_with_yolo_gt(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path = "./yolo_eval_output",
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
    gt_class_names: Optional[List[str] | Dict[int, str]] = None,
    class_id_to_gt_class_id: Optional[Dict[int, int]] = None,
    class_name_to_gt_class_id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    def collect_image_paths_local(images_dir: Path) -> List[Path]:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])

    def get_image_size(image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.size

    def build_gt_class_maps_from_names(
        gt_class_names: Optional[List[str] | Dict[int, str]],
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        if gt_class_names is None:
            return {}, {}
        if isinstance(gt_class_names, dict):
            gt_id_to_name = {int(k): str(v) for k, v in gt_class_names.items()}
        elif isinstance(gt_class_names, list):
            gt_id_to_name = {i: str(v) for i, v in enumerate(gt_class_names)}
        else:
            raise ValueError("gt_class_names deve ser list, dict ou None.")
        gt_name_to_id = {normalize_name(v): k for k, v in gt_id_to_name.items()}
        return gt_name_to_id, gt_id_to_name

    def resolve_class_mapping_inner(model: YOLO) -> Dict[int, int]:
        model_id_to_name, _ = build_model_class_maps(model)
        gt_name_to_id, _ = build_gt_class_maps_from_names(gt_class_names)

        if class_id_to_gt_class_id is not None:
            mapping = {}
            for cls_id, gt_id in class_id_to_gt_class_id.items():
                cls_id = int(cls_id)
                gt_id = int(gt_id)
                if cls_id not in model_id_to_name:
                    raise ValueError(f"Classe do modelo {cls_id} não existe em model.names")
                mapping[cls_id] = gt_id
            return mapping

        if class_name_to_gt_class_id is not None:
            normalized_manual = {normalize_name(k): int(v) for k, v in class_name_to_gt_class_id.items()}
            mapping = {}
            for cls_id, cls_name in model_id_to_name.items():
                norm_name = normalize_name(cls_name)
                if norm_name in normalized_manual:
                    mapping[cls_id] = normalized_manual[norm_name]
            if not mapping:
                raise ValueError("Nenhuma classe do modelo bateu com class_name_to_gt_class_id.")
            return mapping

        if gt_class_names is not None:
            mapping = {}
            for cls_id, cls_name in model_id_to_name.items():
                norm_name = normalize_name(cls_name)
                if norm_name in gt_name_to_id:
                    mapping[cls_id] = gt_name_to_id[norm_name]
            if not mapping:
                raise ValueError(
                    "Não foi possível mapear classes automaticamente. "
                    "Forneça class_id_to_gt_class_id ou class_name_to_gt_class_id."
                )
            return mapping

        print("gt_class_names não informado. Assumindo mapeamento identidade.")
        return {cls_id: cls_id for cls_id in model_id_to_name.keys()}

    def yolo_line_to_coco_bbox(line: str, img_w: int, img_h: int):
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        class_id = int(float(parts[0]))
        x_c, y_c, w, h = map(float, parts[1:])
        x_c *= img_w
        y_c *= img_h
        w *= img_w
        h *= img_h
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        return class_id, [float(x_min), float(y_min), float(w), float(h)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    ensure_pc_backend_for_weights(Path(model_path))
    print("Carregando modelo...")
    model = YOLO(str(model_path))

    print("Convertendo GT YOLO -> COCO...")
    image_paths = collect_image_paths_local(images_dir)
    images: List[dict] = []
    annotations: List[dict] = []
    used_gt_classes = set()
    ann_id = 1
    filename_to_image_id: Dict[str, int] = {}

    for image_id, img_path in enumerate(tqdm(image_paths, desc="Lendo GT YOLO"), start=1):
        img_w, img_h = get_image_size(img_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        images.append(
            {"id": image_id, "file_name": img_path.name, "width": img_w, "height": img_h}
        )
        filename_to_image_id[img_path.name] = image_id
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for line in lines:
                parsed = yolo_line_to_coco_bbox(line, img_w, img_h)
                if parsed is None:
                    continue
                gt_class_id, bbox = parsed
                used_gt_classes.add(gt_class_id)
                x, y, w, h = bbox
                area = max(0.0, w) * max(0.0, h)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(gt_class_id + 1),
                        "bbox": [x, y, w, h],
                        "area": float(area),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    categories = []
    for gt_class_id in sorted(used_gt_classes):
        if gt_class_names is not None:
            if isinstance(gt_class_names, dict):
                cat_name = str(gt_class_names[gt_class_id])
            else:
                cat_name = str(gt_class_names[gt_class_id])
        else:
            cat_name = f"class_{gt_class_id}"
        categories.append({"id": int(gt_class_id + 1), "name": cat_name})

    coco_gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    gt_json_path = output_dir / "gt_from_yolo_coco.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt_dict, f)
    print(f"GT convertido salvo em: {gt_json_path}")

    print("Resolvendo mapeamento de classes...")
    class_mapping = resolve_class_mapping_inner(model)
    print("\nMapeamento final model_class_id -> gt_class_id:")
    names = model.names
    id_to_name = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    for k, v in sorted(class_mapping.items()):
        print(f"  {k} ({id_to_name.get(k, '?')}) -> {v}")

    predictions: List[dict] = []
    skipped_unmapped_class = 0
    for img_path in tqdm(image_paths, desc="Inferindo"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            continue
        image_id = filename_to_image_id[file_name]
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_nms,
            device=device,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) not in class_mapping:
                skipped_unmapped_class += 1
                continue
            gt_class_id = class_mapping[int(cls_id)]
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(gt_class_id + 1),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    print(f"\nDetecções ignoradas por classe não mapeada: {skipped_unmapped_class}")
    print(f"Total de detecções válidas: {len(predictions)}")
    if len(predictions) == 0:
        raise ValueError("Nenhuma predição válida foi gerada.")

    pred_json_path = output_dir / "predictions_coco.json"
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"\nPredições salvas em: {pred_json_path}")

    _, metrics = evaluate_coco(gt_json_path, pred_json_path)
    print("\n=== Métricas COCO ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return {
        "metrics": metrics,
        "gt_json_path": str(gt_json_path),
        "pred_json_path": str(pred_json_path),
        "class_mapping": class_mapping,
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
    }
