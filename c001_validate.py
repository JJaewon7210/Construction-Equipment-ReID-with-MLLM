"""Validation of object detection results using supervision.metrics.

Predictions: lines `class x1 y1 x2 y2 score`.
GT: `track_ID class_ID confidence x1 y1 x2 y2 (absolute values)`
Supports optional class_mapping to align class IDs.
"""
from pathlib import Path
import argparse
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MeanAverageRecall, F1Score, Precision, Recall
import pandas as pd
import sys
from PIL import Image
from collections import defaultdict

def _get_image_path_from_label(label_path: Path) -> Path:
    """Estimate image path for a GT label by replacing 'labels' -> 'images' and .txt -> .jpg.
    Falls back to common extensions and sibling 'images' folders.
    """
    if label_path is None:
        return None

    parts = list(label_path.parts)
    for i, p in enumerate(parts):
        if p.lower() == 'labels':
            parts[i] = 'images'
            candidate = Path(*parts).with_suffix('.jpg')
            if candidate.exists():
                return candidate
            # try common image extensions
            for ext in ('.jpg', '.jpeg', '.png'):
                cand = candidate.with_suffix(ext)
                if cand.exists():
                    return cand
            break

    # Fallback: check sibling 'images' directories up the tree
    for anc in label_path.parents:
        rel = None
        try:
            rel = label_path.relative_to(anc)
        except Exception:
            continue
        candidate = anc / 'images' / rel.with_suffix('.jpg')
        if candidate.exists():
            return candidate
        for ext in ('.jpg', '.jpeg', '.png'):
            cand = anc / 'images' / rel.with_suffix(ext)
            if cand.exists():
                return cand

    # Finally, check for an image with the same name in the label directory
    sibling = label_path.with_suffix('.jpg')
    if sibling.exists():
        return sibling
    for ext in ('.jpg', '.jpeg', '.png'):
        s = label_path.with_suffix(ext)
        if s.exists():
            return s

    return None


def read_ground_truth(file_path: Path, class_mapping=None):
    """Read GT file and return (boxes, img_w, img_h).
    boxes: np.array of shape (N,5) x1,y1,x2,y2,class_id
    If image size is found, YOLO-normalized coords are converted to pixels.
    """

    if file_path is None or not file_path.exists():
        return np.empty((0, 5)), None, None

    # try to get image size from GT label -> image file
    img_path = _get_image_path_from_label(file_path)
    img_w = None
    img_h = None
    if img_path is not None and img_path.exists():
        with Image.open(img_path) as img:
            img_w, img_h = img.size


    boxes = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                raise ValueError(f"GT file {file_path} contains empty line.")

            parts = line.split()
            if len(parts) != 7:
                raise ValueError(f"GT file {file_path} line does not have 7 elements: {line}")


            track_ID = int(parts[0])
            cls = int(float(parts[1]))
            conf = float(parts[2])
            x1, y1, x2, y2 = map(float, parts[3:7])

            # Optional class mapping
            if class_mapping is not None:
                cls = class_mapping.get(cls, cls)

            boxes.append([x1, y1, x2, y2, int(cls)])

    if not boxes:
        return np.empty((0, 5), dtype=float), img_w, img_h

    return np.array(boxes, dtype=float), img_w, img_h


def read_predictions(file_path: Path, img_w: int = None, img_h: int = None, class_mapping=None):
    """Reads prediction (prediction) files and returns an array of (x1,y1,x2,y2,class_id,score).
    If class_mapping is provided, prediction class IDs will be remapped accordingly.
    This is useful when GT and prediction class ID schemes differ.
    """
    preds = []
    if file_path is None or not file_path.exists():
        return np.empty((0, 6))

    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                cls = int(float(parts[0]))
                x1 = float(parts[1]); y1 = float(parts[2]); x2 = float(parts[3]); y2 = float(parts[4])
                score = float(parts[5])
            except Exception:
                continue

            # If normalized (all coords <= 1), scale (image size information required)
            if max(x1, y1, x2, y2) <= 1.0:
                if img_w is not None and img_h is not None:
                    x1 *= img_w; y1 *= img_h; x2 *= img_w; y2 *= img_h
                else:
                    # If image size is unknown, skip this prediction line.
                    print(f"Warning: normalized prediction coordinates found in {file_path} but image size unknown. Skipping this prediction line.")
                    continue

            if class_mapping is not None:
                # Remap prediction class ID according to provided mapping. Use original value if mapping missing.
                cls = class_mapping.get(cls, cls)

            preds.append([x1, y1, x2, y2, int(cls), float(score)])

    if len(preds) == 0:
        return np.empty((0, 6))
    return np.array(preds)


def to_sv_detections(data: np.ndarray, is_prediction: bool = True):
    if data.size == 0:
        return sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([], dtype=int),
                              confidence=np.array([], dtype=float) if is_prediction else None)

    if len(data.shape) == 1:
        expected_length = 6 if is_prediction else 5
        if len(data) != expected_length:
            raise ValueError(f"Expected {expected_length} elements in 1D array, got {len(data)}")
        data = data.reshape(1, -1)

    expected_cols = 6 if is_prediction else 5
    if data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")

    boxes = data[:, :4]
    class_ids = data[:, 4].astype(int)
    if is_prediction:
        confidences = data[:, 5].astype(float)
        return sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidences)
    else:
        return sv.Detections(xyxy=boxes, class_id=class_ids)


def collect_files(pred_dir: Path, gt_dir: Path):
    # Recursively search subdirectories and map files by stem.
    pred_map = {}
    gt_map = {}

    # Recursively find all .txt files inside pred_dir and gt_dir
    for p in pred_dir.rglob('*.txt'):
        stem = p.stem
        if stem in pred_map:
            print(f"Warning: duplicate prediction stem '{stem}' found. Using first: {pred_map[stem]} (ignored {p})")
            continue
        pred_map[stem] = p

    for g in gt_dir.rglob('*.txt'):
        stem = g.stem
        if stem in gt_map:
            # If duplicate GT files exist, use the first and print warning
            print(f"Warning: duplicate GT stem '{stem}' found. Using first: {gt_map[stem]} (ignored {g})")
            continue
        gt_map[stem] = g

    # Use only the intersection of stems
    common_stems = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    pairs = [(pred_map[stem], gt_map[stem], stem) for stem in common_stems]

    print(f"Found {len(pred_map)} prediction files and {len(gt_map)} GT files. Paired by stem: {len(pairs)} items.")
    return pairs


def _iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    a: (Na,4), b: (Nb,4) in xyxy
    return: (Na,Nb) IoU matrix
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a_xyxy.T
    bx1, by1, bx2, by2 = b_xyxy.T
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def _greedy_match_counts(pred_det: sv.Detections, gt_det: sv.Detections, iou_thr: float = 0.5):
    """
    pred_det: sv.Detections with .xyxy (N,4), .class_id (N,), .confidence (N,)
    gt_det:   sv.Detections with .xyxy (M,4), .class_id (M,)
    Return: dicts of TP/FP/FN per class and totals.
    """
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)

    if len(gt_det) == 0 and len(pred_det) == 0:
        return tp_per_class, fp_per_class, fn_per_class

    # sort predictions by confidence desc (standard)
    if pred_det.confidence is not None and len(pred_det) > 0:
        order = np.argsort(-pred_det.confidence)
        p_xyxy = pred_det.xyxy[order]
        p_cls = pred_det.class_id[order]
    else:
        p_xyxy = pred_det.xyxy
        p_cls = pred_det.class_id

    g_xyxy = gt_det.xyxy
    g_cls = gt_det.class_id

    # per-class matching to enforce class-aware evaluation
    classes = set(p_cls.tolist()) | set(g_cls.tolist())
    for c in classes:
        p_idx = np.where(p_cls == c)[0]
        g_idx = np.where(g_cls == c)[0]
        if p_idx.size == 0 and g_idx.size == 0:
            continue
        if p_idx.size == 0 and g_idx.size > 0:
            fn_per_class[c] += int(g_idx.size)
            continue
        if g_idx.size == 0 and p_idx.size > 0:
            fp_per_class[c] += int(p_idx.size)
            continue

        iou = _iou_matrix(p_xyxy[p_idx], g_xyxy[g_idx])
        matched_g = set()
        for i in range(iou.shape[0]):  # predictions already sorted by conf
            # best GT for this pred
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= iou_thr and j not in matched_g:
                tp_per_class[c] += 1
                matched_g.add(j)
            else:
                fp_per_class[c] += 1
        fn_per_class[c] += (iou.shape[1] - len(matched_g))

    return tp_per_class, fp_per_class, fn_per_class


def _sum_counts(counts_dict):
    return sum(counts_dict.values())


def _micro_from_counts(tp_dict, fp_dict, fn_dict):
    TP = _sum_counts(tp_dict)
    FP = _sum_counts(fp_dict)
    FN = _sum_counts(fn_dict)
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
    return {"micro_P": P, "micro_R": R, "micro_F1": F1, "TP": TP, "FP": FP, "FN": FN}


def _class_supports(gts_list: list) -> dict:
    """Return GT box counts (support) per class."""
    sup = defaultdict(int)
    for gt in gts_list:
        for c in gt.class_id.tolist():
            sup[int(c)] += 1
    return dict(sup)


def _weighted_average(values: dict, weights: dict):
    """
    values: {class_id: value}, weights: {class_id: weight}
    returns support-weighted average; if no weights, returns None.
    """
    num = 0.0
    den = 0.0
    for c, v in values.items():
        w = float(weights.get(c, 0))
        if v is None or w <= 0:
            continue
        num += v * w
        den += w
    return (num / den) if den > 0 else None


def compute_metrics(pred_dir: Path, gt_dir: Path):
    pairs = collect_files(pred_dir, gt_dir)

    preds_list = []
    gts_list = []

    total = len(pairs)
    print_step = max(1, total // 100)
    for pair_idx, (pred_path, gt_path, name) in enumerate(pairs):

        # For progress reporting
        if pair_idx % print_step == 0 or pair_idx == total - 1:
            pct = (pair_idx + 1) / total * 100 if total > 0 else 0
            print(f"Processing {pair_idx+1}/{total} ({pct:.1f}%) - {name}  (pred: {pred_path is not None}, gt: {gt_path is not None})")
            
        # Read GT and obtain image size
        if gt_path is not None:
            gt_arr, img_w, img_h = read_ground_truth(gt_path, class_mapping=None)
        else:
            gt_arr, img_w, img_h = np.empty((0, 5)), None, None

        # Pass the image size obtained from GT to predictions
        if pred_path is not None:
            pred_arr = read_predictions(pred_path, img_w=img_w, img_h=img_h, class_mapping=None)
        else:
            pred_arr = np.empty((0, 6))

        preds_list.append(to_sv_detections(pred_arr, is_prediction=True))
        gts_list.append(to_sv_detections(gt_arr, is_prediction=False))

    # Initialize metrics
    map_metric = MeanAveragePrecision()
    mar_metric = MeanAverageRecall()
    f1_metric = F1Score()
    pr_metric = Precision()
    rc_metric = Recall()

    map_res = map_metric.update(preds_list, gts_list).compute()
    mar_res = mar_metric.update(preds_list, gts_list).compute()
    f1_res = f1_metric.update(preds_list, gts_list).compute()
    pr_res = pr_metric.update(preds_list, gts_list).compute()
    rc_res = rc_metric.update(preds_list, gts_list).compute()

    # Collect DataFrames and per-class numbers
    results = {}
    results.update(map_res.to_pandas().iloc[0].to_dict())
    results.update(mar_res.to_pandas().iloc[0].to_dict())
    results.update(f1_res.to_pandas().iloc[0].to_dict())
    results.update(pr_res.to_pandas().iloc[0].to_dict())
    results.update(rc_res.to_pandas().iloc[0].to_dict())

    # ---- IoU threshold-safe column picking (avoid [:,0] magic) ----
    def _closest_idx(thresholds: np.ndarray, target: float) -> int:
        if thresholds is None or len(thresholds) == 0:
            return 0
        return int(np.argmin(np.abs(thresholds - target)))

    # supervision.metrics objects commonly expose iou_thresholds on map_res etc.
    try:
        iou_thrs = map_res.iou_thresholds  # np.array([...])
    except Exception:
        iou_thrs = np.array([0.5])  # fallback
    
    idx_50 = _closest_idx(iou_thrs, 0.5)
    idx_75 = _closest_idx(iou_thrs, 0.75)

    # ---- Per-class metrics with safe indexing ----
    try:
        per_class_f1 = f1_res.f1_per_class[:, idx_50]
    except Exception:
        per_class_f1 = np.array([])
    try:
        per_class_ap = map_res.ap_per_class[:, idx_50]
    except Exception:
        per_class_ap = np.array([])
    try:
        per_class_ar = mar_res.recall_per_class[:, idx_50]
    except Exception:
        per_class_ar = np.array([])
    try:
        per_class_pr = pr_res.precision_per_class[:, idx_50]
    except Exception:
        per_class_pr = np.array([])
    try:
        per_class_rc = rc_res.recall_per_class[:, idx_50]
    except Exception:
        per_class_rc = np.array([])

    # ---- Compute supports BEFORE building per_class_summary ----
    supports = _class_supports(gts_list)  # {class_id: #GT}
    gt_total = sum(supports.values())
    results["GT_total"] = gt_total

    per_class_summary = []
    n_classes = max(len(per_class_f1), len(per_class_ap), len(per_class_ar), len(per_class_pr), len(per_class_rc))
    for i in range(n_classes):
        row = {
            'class_id': i,
            'GT_count': supports.get(i, 0),
            'F1@50': float(per_class_f1[i]) if i < len(per_class_f1) else None,
            'AP@50': float(per_class_ap[i]) if i < len(per_class_ap) else None,
            'AR@50': float(per_class_ar[i]) if i < len(per_class_ar) else None,
            'Precision@50': float(per_class_pr[i]) if i < len(per_class_pr) else None,
            'Recall@50': float(per_class_rc[i]) if i < len(per_class_rc) else None,
        }
        per_class_summary.append(row)

    # ---- Micro counts (tidy comment indentation) ----
    # IoU 0.50
    tp_50 = defaultdict(int); fp_50 = defaultdict(int); fn_50 = defaultdict(int)
    for pred_det, gt_det in zip(preds_list, gts_list):
        tpc, fpc, fnc = _greedy_match_counts(pred_det, gt_det, iou_thr=0.5)
        for k,v in tpc.items(): tp_50[k]+=v
        for k,v in fpc.items(): fp_50[k]+=v
        for k,v in fnc.items(): fn_50[k]+=v
    micro_50 = _micro_from_counts(tp_50, fp_50, fn_50)

    # IoU 0.75
    tp_75 = defaultdict(int); fp_75 = defaultdict(int); fn_75 = defaultdict(int)
    for pred_det, gt_det in zip(preds_list, gts_list):
        tpc, fpc, fnc = _greedy_match_counts(pred_det, gt_det, iou_thr=0.75)
        for k,v in tpc.items(): tp_75[k]+=v
        for k,v in fpc.items(): fp_75[k]+=v
        for k,v in fnc.items(): fn_75[k]+=v
    micro_75 = _micro_from_counts(tp_75, fp_75, fn_75)

    # ---- Macro / wAP ----
    def _nanmean(arr):
        arr = np.array([x for x in arr if x is not None], dtype=float)
        return float(arr.mean()) if arr.size > 0 else None

    macro = {
        'macro_F1@50': _nanmean([r['F1@50'] for r in per_class_summary]),
        'macro_P@50':  _nanmean([r['Precision@50'] for r in per_class_summary]),
        'macro_R@50':  _nanmean([r['Recall@50'] for r in per_class_summary]),
        'macro_AP@50': _nanmean([r['AP@50'] for r in per_class_summary]),
    }

    ap_per_class_dict = {r['class_id']: r['AP@50'] for r in per_class_summary if r['AP@50'] is not None}
    wAP50 = _weighted_average(ap_per_class_dict, supports)

    results.update({
        'micro_P@50':  micro_50['micro_P'],
        'micro_R@50':  micro_50['micro_R'],
        'micro_F1@50': micro_50['micro_F1'],
        'micro_TP@50': micro_50['TP'],
        'micro_FP@50': micro_50['FP'],
        'micro_FN@50': micro_50['FN'],

        'micro_P@75':  micro_75['micro_P'],
        'micro_R@75':  micro_75['micro_R'],
        'micro_F1@75': micro_75['micro_F1'],
        'micro_TP@75': micro_75['TP'],
        'micro_FP@75': micro_75['FP'],
        'micro_FN@75': micro_75['FN'],

        'macro_F1@50': macro['macro_F1@50'],
        'macro_P@50':  macro['macro_P@50'],
        'macro_R@50':  macro['macro_R@50'],
        'macro_AP@50': macro['macro_AP@50'],

        'wAP@50': wAP50,
        'GT_total': gt_total,
    })
    
    return results, per_class_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default= 'detections/Busan_01/', 
                        help='Directory with prediction .txt files')
    parser.add_argument('--gt_dir', default= 'data/Busan_01/labels', 
                        help='Directory with ground-truth .txt files')
    parser.add_argument('--out', default=None, 
                        help='Optional CSV path to write per-class metrics')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    if not pred_dir.exists():
        print(f"Prediction directory not found: {pred_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"GT directory not found: {gt_dir}")
        sys.exit(1)

    results, per_class = compute_metrics(pred_dir, gt_dir)

    print("Overall metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nPer-class metrics:")
    for row in per_class:
        print(row)

    if args.out:
        df = pd.DataFrame(per_class)
        df.to_csv(args.out, index=False)
        print(f"Per-class metrics written to {args.out}")

if __name__ == '__main__':
    main()
