"""Inferência YOLO no PC + COCOeval (GT em JSON COCO ou labels YOLO)."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
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


def detect_tflite_input_hw(model_path: Path) -> Optional[Tuple[int, int]]:
    """
    Lê o tensor de entrada do .tflite e devolve (H, W) quando for estático.
    Útil porque alguns exports fixam 512x512 (ou NCHW/NHWC) e o `imgsz` do YAML
    pode estar em 640 — no PC isso pode degradar resultados ou falhar.
    """
    if model_path.suffix.lower() != ".tflite":
        return None
    ensure_pc_backend_for_weights(model_path)
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]

        interpreter = Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf  # type: ignore[import-untyped]

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()
    if not details:
        return None
    shape = details[0].get("shape")
    if shape is None:
        return None
    # shape pode ser np.ndarray ou list
    try:
        dims = [int(x) for x in list(shape)]
    except Exception:
        return None
    if len(dims) < 3:
        return None
    # Heurística comum: NCHW (1,3,H,W) ou NHWC (1,H,W,3)
    if dims[1] in (1, 3) and dims[-1] not in (1, 3):
        h, w = dims[2], dims[3]
    elif dims[-1] in (1, 3) and dims[1] not in (1, 3):
        h, w = dims[1], dims[2]
    else:
        # fallback: tenta os dois últimos inteiros > 3
        spatial = [d for d in dims if d > 3]
        if len(spatial) >= 2:
            h, w = spatial[-2], spatial[-1]
        else:
            return None
    if h <= 0 or w <= 0:
        return None
    return h, w


def resolve_effective_imgsz(model_path: Path, requested_imgsz: int) -> int:
    """
    Para TFLite com entrada espacial fixa, alinha `imgsz` ao H=W do tensor.
    Para outros formatos, mantém o pedido.
    """
    hw = detect_tflite_input_hw(model_path)
    if hw is None:
        return int(requested_imgsz)
    h, w = hw
    if h != w:
        print(f"Aviso: entrada TFLite não quadrada ({h}x{w}); usando max(H,W) como imgsz.")
        aligned = max(h, w)
    else:
        aligned = h
    if int(aligned) != int(requested_imgsz):
        print(f"Aviso: ajustando imgsz de {requested_imgsz} -> {aligned} com base no tensor de entrada do TFLite.")
    return int(aligned)


def _extract_expected_imgsz_from_error(error: Exception) -> Optional[int]:
    """
    Tenta extrair o tamanho esperado pelo backend TFLite a partir da mensagem de erro.
    Exemplo: "Got 640 but expected 512 for dimension 1 of input 0."
    """
    msg = str(error)
    m = re.search(r"expected\s+(\d+)\s+for\s+dimension\s+1", msg)
    if m:
        return int(m.group(1))
    m = re.search(r"dimension mismatch.*expected\s+(\d+)", msg, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _predict_with_imgsz_fallback(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    conf_thres: float,
    iou_nms: float,
    device: Optional[str],
    max_det: Optional[int] = None,
) -> Tuple[Any, int]:
    """
    Executa inferência e, se houver mismatch de dimensão típico de TFLite, adapta
    o `imgsz` automaticamente e repete a chamada.
    """
    def _predict_once(sz: int):
        kwargs: Dict[str, Any] = {}
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        return model.predict(
            source=str(image_path),
            imgsz=int(sz),
            conf=conf_thres,
            iou=iou_nms,
            device=device,
            verbose=False,
            **kwargs,
        )

    try:
        results = _predict_once(imgsz)
        return results, imgsz
    except Exception as e:
        expected = _extract_expected_imgsz_from_error(e)
        if expected is None or expected == imgsz:
            raise
        print(
            f"Aviso: {image_path.name} falhou com imgsz={imgsz}. "
            f"Modelo espera imgsz={expected}; repetindo inferência com fallback."
        )
        results = _predict_once(expected)
        return results, int(expected)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _build_timing_stats(inference_times_ms: List[float]) -> Dict[str, Optional[float]]:
    if not inference_times_ms:
        return {
            "num_images_with_time": 0,
            "avg_inference_time_ms": None,
            "min_inference_time_ms": None,
            "max_inference_time_ms": None,
            "median_inference_time_ms": None,
            "p95_inference_time_ms": None,
            "p99_inference_time_ms": None,
            "std_inference_time_ms": None,
        }
    return {
        "num_images_with_time": len(inference_times_ms),
        "avg_inference_time_ms": float(sum(inference_times_ms) / len(inference_times_ms)),
        "min_inference_time_ms": float(min(inference_times_ms)),
        "max_inference_time_ms": float(max(inference_times_ms)),
        "median_inference_time_ms": _percentile(inference_times_ms, 50.0),
        "p95_inference_time_ms": _percentile(inference_times_ms, 95.0),
        "p99_inference_time_ms": _percentile(inference_times_ms, 99.0),
        "std_inference_time_ms": float(statistics.pstdev(inference_times_ms)),
    }


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
    max_det: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Optional[float]]]:
    filename_to_image_id = build_filename_to_image_id(coco_gt)
    predictions: List[dict] = []
    skipped_no_gt = 0
    skipped_unmapped_class = 0
    inference_times_ms: List[float] = []
    current_imgsz = imgsz
    for img_path in tqdm(image_paths, desc="Inferência"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            skipped_no_gt += 1
            continue
        image_id = filename_to_image_id[file_name]
        t0 = time.perf_counter()
        results, current_imgsz = _predict_with_imgsz_fallback(
            model=model,
            image_path=img_path,
            imgsz=current_imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
        )
        inference_times_ms.append((time.perf_counter() - t0) * 1000.0)
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
    timing = _build_timing_stats(inference_times_ms)
    return predictions, timing


def run_yolo_predictions_coco80_mapping(
    model: YOLO,
    image_paths: List[Path],
    coco_gt: COCO,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
    max_det: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Optional[float]]]:
    """Para modelos treinados no conjunto COCO com 80 classes na ordem padrão YOLO."""
    filename_to_image_id = build_filename_to_image_id(coco_gt)
    predictions: List[dict] = []
    inference_times_ms: List[float] = []
    current_imgsz = imgsz
    for img_path in tqdm(image_paths, desc="Inferência (COCO80)"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            continue
        image_id = filename_to_image_id[file_name]
        t0 = time.perf_counter()
        results, current_imgsz = _predict_with_imgsz_fallback(
            model=model,
            image_path=img_path,
            imgsz=current_imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
        )
        inference_times_ms.append((time.perf_counter() - t0) * 1000.0)
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
    timing = _build_timing_stats(inference_times_ms)
    return predictions, timing


def inference_and_evaluate_coco_gt(
    model_path: str | Path,
    val_images_dir: str | Path,
    ann_file: str | Path,
    out_predictions_json: str | Path,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
    max_det: Optional[int] = None,
    max_images: Optional[int] = None,
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
    imgsz = resolve_effective_imgsz(model_path, int(imgsz))
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
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]
    print(f"\nTotal de imagens encontradas: {len(image_paths)}")

    if class_mapping_mode == "coco80":
        predictions, timing = run_yolo_predictions_coco80_mapping(
            model=model,
            image_paths=image_paths,
            coco_gt=coco_gt,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
        )
    else:
        predictions, timing = run_yolo_predictions_to_coco_json(
            model=model,
            image_paths=image_paths,
            coco_gt=coco_gt,
            class_mapping=class_mapping,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
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
    if timing.get("avg_inference_time_ms") is not None:
        print(f"Tempo médio de inferência (ms): {timing['avg_inference_time_ms']:.3f}")
    return {
        "metrics": metrics,
        "timing": timing,
        "predictions_path": str(out_predictions_json),
        "num_predictions": len(predictions),
    }


def evaluate_yolo_with_yolo_gt(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path = "./yolo_eval_output",
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_nms: float = 0.7,
    device: Optional[str] = None,
    max_det: Optional[int] = None,
    max_images: Optional[int] = None,
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
    imgsz = resolve_effective_imgsz(Path(model_path), int(imgsz))

    print("Convertendo GT YOLO -> COCO...")
    image_paths = collect_image_paths_local(images_dir)
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]
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
    inference_times_ms: List[float] = []
    current_imgsz = imgsz
    for img_path in tqdm(image_paths, desc="Inferindo"):
        file_name = img_path.name
        if file_name not in filename_to_image_id:
            continue
        image_id = filename_to_image_id[file_name]
        t0 = time.perf_counter()
        results, current_imgsz = _predict_with_imgsz_fallback(
            model=model,
            image_path=img_path,
            imgsz=current_imgsz,
            conf_thres=conf_thres,
            iou_nms=iou_nms,
            device=device,
            max_det=max_det,
        )
        inference_times_ms.append((time.perf_counter() - t0) * 1000.0)
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
    timing = _build_timing_stats(inference_times_ms)
    if timing.get("avg_inference_time_ms") is not None:
        print(f"Tempo médio de inferência (ms): {timing['avg_inference_time_ms']:.3f}")

    return {
        "metrics": metrics,
        "timing": timing,
        "gt_json_path": str(gt_json_path),
        "pred_json_path": str(pred_json_path),
        "class_mapping": class_mapping,
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
    }
