"""
MOT evaluation with TrackEval (HOTA / DetA / AssA / IDF1 / MOTA)

- Input format (per-frame TXT for both GT and predictions):
    line: <tid> <cid> <conf> <x1> <y1> <x2> <y2>
- Processing:
    * Builds a MOTChallenge-style workspace for a single sequence that
      spans ALL frames found under --gt-root (frame_*.txt).
    * Optionally also generates per-class sequences.
    * Runs TrackEval (MotChallenge2DBox) once over these sequences.
- Output:
    * <out_xlsx> with two sheets:
        - metrics_all: TrackEval detailed metrics per sequence
        - counts: basic counts per sequence (frames / GT boxes / HYP boxes)
    * TrackEval native outputs under the workspace tracker folder.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
# ---- Numpy 2.x compatibility for TrackEval (uses np.float/np.int) ----
if not hasattr(np, "float"): np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):   np.int   = int    # type: ignore[attr-defined]
import pandas as pd

# TrackEval
from trackeval import Evaluator
from trackeval import datasets as te_datasets
from trackeval import metrics as te_metrics


# ---------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------

# Map your integer class ids to human-friendly names (used in per-class sequences)
CLASS_NAME_MAP = {0: "dumptruck", 1: "excavator"}

# Frame filename pattern
FRAME_RE = re.compile(r"frame_(\d+)\.txt$", re.IGNORECASE)


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------

def stem_from_path(p: Path) -> str:
    """Return the numeric frame stem (e.g., '000123') from 'frame_000123.txt'."""
    m = FRAME_RE.search(p.name)
    if not m:
        raise ValueError(f"Invalid frame filename: {p}")
    return m.group(1)


def read_txt_boxes(txt_path: Path) -> List[dict]:
    """
    Parse one frame TXT into a list of boxes with keys:
        id (str), cls (int), conf (float), xyxy (np.ndarray shape [4])
    Returns [] if file missing.
    """
    boxes: List[dict] = []
    if not txt_path.exists():
        return boxes

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            tid, cid, conf = parts[0], int(parts[1]), float(parts[2])
            x1, y1, x2, y2 = map(float, parts[3:7])
            boxes.append(
                {"id": str(tid), "cls": cid, "conf": conf, "xyxy": np.array([x1, y1, x2, y2], float)}
            )
    return boxes


def dedup_by_id_keep_max_conf(entries: List[dict]) -> List[dict]:
    """Within a frame, keep only the highest-confidence box for each track id."""
    best: Dict[str, dict] = {}
    for e in entries:
        tid = e["id"]
        if (tid not in best) or (e["conf"] > best[tid]["conf"]):
            best[tid] = e
    return list(best.values())


def collect_pred_frame_map(pred_root: Path, part_glob: str) -> Dict[str, Path]:
    """
    Aggregate all predicted frame_*.txt across part folders.
    Returns mapping: 'frame_stem' -> prediction txt path
    """
    frame_map: Dict[str, Path] = {}

    part_dirs = sorted(
        glob.glob(str(pred_root / part_glob)),
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)],
    )
    for d in part_dirs:
        for p in sorted(Path(d).glob("frame_*.txt")):
            try:
                stem = stem_from_path(p)
            except ValueError:
                continue
            if stem not in frame_map:
                frame_map[stem] = p

    # Also include files directly under pred_root (if any)
    for p in sorted(pred_root.glob("frame_*.txt")):
        try:
            stem = stem_from_path(p)
        except ValueError:
            continue
        if stem not in frame_map:
            frame_map[stem] = p

    return frame_map


# ---------------------------------------------------------------------
# TrackEval workspace construction
# ---------------------------------------------------------------------

def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _ensure_int_id(id_str: str, id_map: Dict[str, int]) -> int:
    """Map arbitrary string track IDs to 1..K integers as required by MOTChallenge."""
    if id_str not in id_map:
        id_map[id_str] = len(id_map) + 1
    return id_map[id_str]


def _write_seqinfo_ini(
    seq_dir: Path,
    seq_name: str,
    seq_len: int,
    im_w: int = 1920,
    im_h: int = 1080,
    frame_rate: int = 30,
) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    ini = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={seq_len}\n"
        f"imWidth={im_w}\n"
        f"imHeight={im_h}\n"
        "imExt=.jpg\n"
    )
    (seq_dir / "seqinfo.ini").write_text(ini, encoding="utf-8")


def build_trackeval_files_for_sequence(
    gt_workspace_root: Path,
    trk_workspace_dir: Path,
    seq_name: str,
    frames_sorted: List[int],
    gt_root: Path,
    pred_map: Dict[str, Path],
    class_filter: int | None,
    pad_width: int,
) -> Tuple[int, int, int]:
    """
    Create MOTChallenge files for one sequence.
    Returns (seq_len, #GT lines, #TRK lines).
    """
    # Remap original frame numbers to contiguous [1..K]
    frame_index = {fr: i + 1 for i, fr in enumerate(frames_sorted)}
    seq_len = len(frames_sorted)

    # Paths (SKIP_SPLIT_FOL=True => put things directly under mot_challenge)
    gt_dir = gt_workspace_root / seq_name / "gt"
    trk_workspace_dir = trk_workspace_dir
    gt_dir.mkdir(parents=True, exist_ok=True)
    trk_workspace_dir.mkdir(parents=True, exist_ok=True)

    gt_lines: List[str] = []
    tr_lines: List[str] = []

    gt_id_map: Dict[str, int] = {}
    trk_id_map: Dict[str, int] = {}

    for fr in frames_sorted:
        stem = str(fr).zfill(pad_width)
        mot_t = frame_index[fr]

        gt_path = gt_root / f"frame_{stem}.txt"
        pr_path = pred_map.get(stem, None)

        gt_entries = read_txt_boxes(gt_path)
        pr_entries = dedup_by_id_keep_max_conf(read_txt_boxes(pr_path)) if pr_path else []

        if class_filter is not None:
            gt_entries = [e for e in gt_entries if e["cls"] == class_filter]
            pr_entries = [e for e in pr_entries if e["cls"] == class_filter]

        # GT rows: mark=1, class=1 (fixed), last two cols -1
        for e in gt_entries:
            x, y, w, h = _xyxy_to_xywh(*e["xyxy"])
            iid = _ensure_int_id(e["id"], gt_id_map)
            gt_lines.append(f"{mot_t},{iid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,-1,-1\n")

        # Tracker rows: confidence from prediction, class=1
        for e in pr_entries:
            x, y, w, h = _xyxy_to_xywh(*e["xyxy"])
            iid = _ensure_int_id(e["id"], trk_id_map)
            tr_lines.append(f"{mot_t},{iid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{e['conf']:.6f},1,-1,-1\n")

    # Write files
    _write_seqinfo_ini(gt_workspace_root / seq_name, seq_name, seq_len)
    (gt_dir / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")

    # Each sequence's tracker output is a single file named <seq_name>.txt
    (trk_workspace_dir / f"{seq_name}.txt").write_text("".join(tr_lines), encoding="utf-8")

    return seq_len, len(gt_lines), len(tr_lines)


def build_seqmap_file(gt_workspace_root: Path, split_name: str, seq_names: List[str]) -> Path:
    seqmap_dir = gt_workspace_root / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    lines = ["name\n"] + [f"{s}\n" for s in seq_names]
    path = seqmap_dir / f"{split_name}.txt"
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# TrackEval runner
# ---------------------------------------------------------------------

def run_trackeval(
    workspace_gt: Path,
    workspace_trk_data_dir: Path,
    tracker_name: str,
    seqmap_file: Path,
    use_parallel: bool = False,
    n_cores: int = 4,
):
    """
    Configure and run TrackEval (MotChallenge2DBox).
    TrackEval will place outputs under: <workspace>/data/trackers/mot_challenge/<tracker_name>/
    """
    ws_trk_data = Path(workspace_trk_data_dir)  # .../trackers/mot_challenge/<tracker_name>/data

    ds_cfg = te_datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(workspace_gt),
        "TRACKERS_FOLDER": str(ws_trk_data.parent.parent),  # .../trackers/mot_challenge
        "TRACKERS_TO_EVAL": [tracker_name],
        "TRACKER_SUB_FOLDER": ws_trk_data.name,             # "data"
        "OUTPUT_FOLDER": str(ws_trk_data.parent),
        "SPLIT_TO_EVAL": "train",
        "BENCHMARK": "MOT17",
        "SEQMAP_FILE": str(seqmap_file),
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False,
        # We set class column to 1 for all rows, so default evaluation class is fine.
    })

    ev_cfg = {
        "USE_PARALLEL": use_parallel,
        "NUM_PARALLEL_CORES": n_cores,
        "PRINT_CONFIG": True,
        "OUTPUT_DETAILED": True,
    }

    evaluator = Evaluator(ev_cfg)
    dataset_list = [te_datasets.MotChallenge2DBox(ds_cfg)]
    metrics_list = [te_metrics.HOTA(), te_metrics.CLEAR(), te_metrics.Identity()]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TrackEval runner")
    p.add_argument("--pred-root", type=str,
                   default="tracked_outputs/Busan_01/tracked_boosttrack+ReID(Youtube)+gpt-5-mini",
                   help="Folder containing prediction parts (e.g., tracked_results_partXXX/frame_*.txt).")
    p.add_argument("--pred-part-glob", type=str, default="tracked_results_part*",
                   help="Glob under --pred-root to locate frame_*.txt files.")
    p.add_argument("--gt-root", type=str, default="data/Busan_01/labels",
                   help="Folder with GT frame_*.txt files.")
    p.add_argument("--seq-name", type=str, default="Busan_01",
                   help="Base name for the sequence.")
    p.add_argument("--workspace", type=str, default="trackeval_workspace/Busan_01",
                   help="Temporary workspace to build MOTChallenge inputs/outputs.")
    p.add_argument("--out-xlsx", type=str,
                   default="tracked_outputs/Busan_01/tracked_boosttrack+ReID(Youtube)+gpt-5-mini/trackeval_metrics_all.xlsx",
                   help="Excel file to save metrics and counts.")
    p.add_argument("--include-per-class", action="store_true",
                   help="Also produce per-class sequences (one per class id in CLASS_NAME_MAP).")
    p.add_argument("--parallel", action="store_true", help="Enable TrackEval multiprocessing.")
    p.add_argument("--cores", type=int, default=4, help="Num cores for TrackEval if --parallel is set.")
    return p.parse_args()

def main():
    args = parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    assert pred_root.exists(), f"--pred-root not found: {pred_root}"
    assert gt_root.exists(), f"--gt-root not found: {gt_root}"

    # Index predictions
    print(f"[INFO] Indexing predictions under: {pred_root} (glob='{args.pred_part_glob}')")
    pred_map = collect_pred_frame_map(pred_root, args.pred_part_glob)
    print(f"[INFO] #prediction frames indexed: {len(pred_map)}")

    # Determine available GT frames and zero-pad width
    gt_files = sorted(gt_root.glob("frame_*.txt"), key=lambda p: int(stem_from_path(p)))
    if not gt_files:
        raise RuntimeError("No GT frame_*.txt found.")
    pad_width = len(stem_from_path(gt_files[0]))
    all_frames = [int(stem_from_path(p)) for p in gt_files]
    print(f"[INFO] #GT frames: {len(all_frames)} (min={min(all_frames)}, max={max(all_frames)})")

    # Build workspace paths
    ws_root = Path(args.workspace)
    ws_gt = ws_root / "data" / "gt" / "mot_challenge"
    ws_trk_root = ws_root / "data" / "trackers" / "mot_challenge"
    tracker_name = pred_root.name or "tracker"
    ws_trk_data = ws_trk_root / tracker_name / "data"
    ws_out_dir = ws_trk_root / tracker_name  # TrackEval output root

    # Build sequences: 1) overall ALL_FRAMES, 2) optional per-class
    seq_names: List[str] = []
    counts_rows: List[dict] = []

    # Overall sequence
    seq_overall = f"{args.seq_name}_ALL_FRAMES"
    seq_len, n_gt, n_tr = build_trackeval_files_for_sequence(
        gt_workspace_root=ws_gt,
        trk_workspace_dir=ws_trk_data,
        seq_name=seq_overall,
        frames_sorted=sorted(all_frames),
        gt_root=gt_root,
        pred_map=pred_map,
        class_filter=None,
        pad_width=pad_width,
    )
    seq_names.append(seq_overall)
    counts_rows.append({
        "sequence": seq_overall,
        "kind": "overall",
        "num_frames": seq_len,
        "gt_boxes": n_gt,
        "hyp_boxes": n_tr,
    })
    print(f"[INFO] Built sequence '{seq_overall}'  frames={seq_len}, GT={n_gt}, HYP={n_tr}")

    # Per-class sequences
    if args.include_per_class:
        for cid, cname in CLASS_NAME_MAP.items():
            seq_c = f"{args.seq_name}_ALL_FRAMES_cls{cid}_{cname}"
            seq_len_c, n_gt_c, n_tr_c = build_trackeval_files_for_sequence(
                gt_workspace_root=ws_gt,
                trk_workspace_dir=ws_trk_data,
                seq_name=seq_c,
                frames_sorted=sorted(all_frames),
                gt_root=gt_root,
                pred_map=pred_map,
                class_filter=cid,
                pad_width=pad_width,
            )
            seq_names.append(seq_c)
            counts_rows.append({
                "sequence": seq_c,
                "kind": f"class_{cid}",
                "num_frames": seq_len_c,
                "gt_boxes": n_gt_c,
                "hyp_boxes": n_tr_c,
            })
            print(f"[INFO] Built sequence '{seq_c}'  frames={seq_len_c}, GT={n_gt_c}, HYP={n_tr_c}")

    # Build seqmap for TrackEval
    seqmap_file = build_seqmap_file(ws_gt, "train", seq_names)

    # Run TrackEval
    print(f"[INFO] Running TrackEval on {len(seq_names)} sequence(s)...")
    _ = run_trackeval(
        workspace_gt=ws_gt,
        workspace_trk_data_dir=ws_trk_data,
        tracker_name=tracker_name,
        seqmap_file=seqmap_file,
        use_parallel=args.parallel,
        n_cores=args.cores,
    )

    # Locate TrackEval detailed CSV (e.g., pedestrian_detailed.csv)
    detailed_csv_candidates = sorted(ws_out_dir.rglob("*_detailed.csv"))
    if not detailed_csv_candidates:
        raise FileNotFoundError(f"No *_detailed.csv produced under {ws_out_dir}")
    detailed_csv = detailed_csv_candidates[0]
    print(f"[INFO] Parsing metrics from: {detailed_csv}")

    det_df = pd.read_csv(detailed_csv)

    # Normalize the sequence column name to 'seq'
    if "seq" not in det_df.columns:
        for c in ("Sequence", "SEQ", "sequence"):
            if c in det_df.columns:
                det_df = det_df.rename(columns={c: "seq"})
                break
    if "seq" not in det_df.columns:
        raise KeyError(f"'seq' (or Sequence) column not found in {detailed_csv.name}")

    # Add a simple 'kind' label inferred from the sequence name
    def infer_kind(s: str) -> str:
        s = str(s)
        if "_cls" in s:
            m = re.search(r"_cls(\d+)_", s)
            return f"class_{m.group(1)}" if m else "class_unknown"
        return "overall" if s.endswith("_ALL_FRAMES") else "unknown"

    det_df["kind"] = det_df["seq"].apply(infer_kind)

    # Reorder columns (seq, kind first)
    cols = det_df.columns.tolist()
    ordered = ["seq", "kind"] + [c for c in cols if c not in ("seq", "kind")]
    det_df = det_df[ordered]

    # Save Excel (metrics + counts)
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        det_df.to_excel(w, sheet_name="metrics_all", index=False)
        pd.DataFrame(counts_rows).to_excel(w, sheet_name="counts", index=False)

    print(f"[INFO] Saved Excel: {out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
