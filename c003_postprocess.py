"""
Postprocess tracking outputs frame-by-frame, call the adjudicator for each newly appearing ID,
apply a remap table based on adjudicator decisions, and write remapped final TXT outputs.
"""
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2

from c003_gpt_adjudicator import NewIdEvent, CandidateTrack, adjudicate_new_id

# =============================================================================
# Configuration
# =============================================================================
FRAMES_DIR = "data/Busan_01/images"  # folder with original frames

TRACKS_ROOTS = [
    f"tracked_outputs/Busan_01/tracked_boosttrack+ReID(Youtube)/tracked_results_part{str(i).zfill(3)}"
    for i in range(4)
]
OUTPUT_ROOT = "tracked_outputs/Busan_01/tracked_boosttrack+ReID(Youtube)+gpt-5-mini/"

CLASSES = ["truck", "excavator"]
FPS = 30.0
WINDOW_SEC = 30.0
MAX_CANDIDATES = 10
WARMUP_FRAMES = 1

# Save options
SAVE_FRAME_OVERLAY = False
OVERLAY_SUBDIR = "overlay"
GPT_OVERLAY_SUBDIR = "gpt_overlay"
FINAL_TXT_SUBDIR = "tracked_results_part"

# Global palette (fixed color per final ID)
GLOBAL_PALETTE = [
    ("red", (0, 0, 255)),
    ("blue", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("purple", (128, 0, 128)),
    ("lime", (0, 255, 128)),
    ("gray", (180, 180, 180)),
    ("orange", (0, 165, 255)),
]

# Candidate overlay colors
GPT_CANDIDATE_COLORS = [
    ("green", (0, 255, 0)),
    ("red", (0, 0, 255)),
    ("blue", (255, 0, 0)),
    ("purple", (128, 0, 128)),
    ("lime", (0, 255, 128)),
]

# Special color for NEW object boxes (query object)
COLOR_PINK = ("pink", (203, 192, 255))

# =============================================================================
# Utilities
# =============================================================================

def _sanitize_stem(stem: str) -> str:
    """Make a file-stem safe for filenames: keep alnum, underscore, hyphen."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', stem)


def _fname_new(stem: str, new_tid: int, frame_idx: int) -> str:
    return f"new.nid{int(new_tid):05d}.f{int(frame_idx):06d}.{_sanitize_stem(stem)}.jpg"


def _fname_cand(stem: str, new_tid: int, cand_final: int, rank: int, where: str, frame_idx: int, delta_t_sec: float) -> str:
    dt = int(round(delta_t_sec))
    return (
        f"cand{int(rank):02d}.nid{int(new_tid):05d}.cid{int(cand_final):05d}."
        f"{where}.f{int(frame_idx):06d}.dt{dt}s.{_sanitize_stem(stem)}.jpg"
    )


def _boxes_for_tid(raw_tracks: List[Tuple], tid: int) -> List[Tuple[int, int, int, int, int]]:
    """Return all boxes in raw_tracks for the given raw tid as integer coordinates.
    raw_tracks expected format per row: (x1,y1,x2,y2,tid,conf,cid)"""
    out = []
    for t in raw_tracks:
        try:
            if int(t[4]) == int(tid):
                x1, y1, x2, y2 = map(int, (t[0], t[1], t[2], t[3]))
                cid = int(t[6])
                out.append((x1, y1, x2, y2, cid))
        except Exception:
            continue
    return out


def _unique_boxes(boxes_xyxy: List[Tuple]) -> List[Tuple]:
    """Remove duplicate boxes by coordinates while preserving order."""
    seen = set()
    out = []
    for b in boxes_xyxy:
        key = (b[0], b[1], b[2], b[3])
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def _area_xyxy(b: Tuple) -> int:
    x1, y1, x2, y2 = b[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _main_and_extras(boxes_xyxy: List[Tuple]):
    """Return the largest box as main and the rest as extras."""
    if not boxes_xyxy:
        return None, []
    boxes_sorted = sorted(boxes_xyxy, key=lambda b: _area_xyxy(b), reverse=True)
    main_box = boxes_sorted[0]
    extras = boxes_sorted[1:]
    return main_box, extras


def color_for_id(tid: int):
    name, bgr = GLOBAL_PALETTE[tid % len(GLOBAL_PALETTE)]
    return name, bgr


def read_tracks_txt(txt_path: Optional[Path]) -> List[Tuple[float, float, float, float, int, float, int]]:
    """Read a tracking TXT and return list of (x1,y1,x2,y2,tid,conf,cid)."""
    if not txt_path or not txt_path.exists():
        return []
    out = []
    with txt_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            s = line.strip().split()
            if len(s) < 7:
                continue
            tid = int(float(s[0])); cid = int(float(s[1])); conf = float(s[2])
            x1, y1, x2, y2 = map(float, s[3:7])
            out.append((x1, y1, x2, y2, tid, conf, cid))
    return out


def normalize_stem(stem: str) -> str:
    return stem[6:] if stem.startswith("frame_") else stem


def build_stem_to_txt_map_multi(track_roots: List[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for root in track_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.txt"):
            key = normalize_stem(p.stem)
            old = mapping.get(key)
            if old and str(old) != str(p):
                print(f"[INFO] duplicate key '{key}' -> override:\n  old={old}\n  new={p}")
            mapping[key] = p
    return mapping


def put_text_with_outline(img, text, org, font_scale, color_bgr, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)


def draw_id_box(canvas, box, tid, color_override=None, draw_tid=True):
    x1, y1, x2, y2 = map(int, box)
    _, color_bgr = color_for_id(int(tid)) if color_override is None else color_override
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 4)

    bw, bh = x2 - x1, y2 - y1
    big_scale = max(0.9, min(3.0, 0.004 * max(bw, bh)))
    big_thick = max(2, int(round(big_scale * 2.2)))
    id_text = f"{tid}"
    (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, big_scale, big_thick)
    cx = x1 + bw // 2; cy = y1 + bh // 2
    org_big = (max(0, cx - tw // 2), max(th + 2, cy + th // 2))
    if draw_tid:
        put_text_with_outline(canvas, id_text, org_big, big_scale, color_bgr, big_thick)


def save_frame_overlay(img, tracks, classes, save_path: Path):
    vis = img.copy()
    for trk in tracks:
        x1, y1, x2, y2, tid, conf, cid, *rest = trk
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        _, color_bgr = color_for_id(int(tid))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 3)
        bw, bh = x2 - x1, y2 - y1
        big_scale = max(0.7, min(2.5, 0.003 * max(bw, bh)))
        big_thick = max(2, int(round(big_scale * 2.0)))
        id_text = f"{tid}"
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, big_scale, big_thick)
        cx = x1 + bw // 2
        cy = y1 + bh // 2
        org_big = (max(0, cx - tw // 2), max(th + 2, cy + th // 2))
        put_text_with_outline(vis, id_text, org_big, big_scale, color_bgr, big_thick)
        cls_name = classes[cid] if 0 <= cid < len(classes) else f"cls{cid}"
        small = f"{cls_name}-{tid} {conf:.2f}"
        put_text_with_outline(vis, small, (x1, max(0, y1 - 6)), 0.6, color_bgr, 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)

# =============================================================================
# GPT multi-frame overlay builder
# =============================================================================

def _load_frame_and_tracks(frame_files: List[Path], stem2txt: Dict[str, Path], frame_idx: int):
    if frame_idx < 0 or frame_idx >= len(frame_files):
        return None, None, None
    f = frame_files[frame_idx]
    img = cv2.imread(str(f))
    if img is None:
        return None, None, None
    stem_key = normalize_stem(f.stem)
    txt = stem2txt.get(stem_key)
    raw_tracks = read_tracks_txt(txt) if txt else []
    return f.stem, img, raw_tracks


def build_gpt_overlays_multi(
    cur_idx: int,
    frame_files: List[Path],
    stem2txt: Dict[str, Path],
    img_cur, stem_cur,
    new_tid, new_cid, new_box,
    raw_tracks_cur,
    cand_rows: List[Tuple],
    classes, save_dir: Path, max_cands: int
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Assign colors to candidates deterministically by index
    color_assign: Dict[int, Tuple[str, Tuple[int, int, int]]] = {}
    for i, cand in enumerate(cand_rows):
        if not (isinstance(cand, tuple) and len(cand) == 5):
            raise ValueError("Invalid candidate format")
        cand_final = int(cand[0])
        base_color = GPT_CANDIDATE_COLORS[min(i, len(GPT_CANDIDATE_COLORS) - 1)]
        color_assign[cand_final] = base_color

    overlays = []

    # NEW-only image: draw the NEW object with pink box
    canvas_new = img_cur.copy()
    new_boxes_full = _boxes_for_tid(raw_tracks_cur, new_tid)
    if not new_boxes_full:
        x1, y1, x2, y2 = map(int, new_box)
        new_boxes_full = [(x1, y1, x2, y2, int(new_cid))]
    new_boxes_full = _unique_boxes(new_boxes_full)
    main_new, extra_new = _main_and_extras(new_boxes_full)

    # Draw all boxes for the new object in pink, only on the NEW-only image
    for b in sorted(new_boxes_full, key=_area_xyxy, reverse=True):
        draw_id_box(canvas_new, b[:4], new_tid, color_override=COLOR_PINK, draw_tid=False)

    out_new = save_dir / _fname_new(stem_cur, new_tid, cur_idx)
    cv2.imwrite(str(out_new), canvas_new)
    overlays.append({
        "path": str(out_new),
        "participants": [{
            "role": "new",
            "alias": "new",
            "tid": int(new_tid),
            "color": "pink",
            "class_name": classes[new_cid] if 0 <= new_cid < len(classes) else f"cls{new_cid}",
            "bbox": list(map(int, (main_new[:4] if main_new else new_boxes_full[0][:4]))),
            "extra_bboxes": [list(map(int, b[:4])) for b in (extra_new or [])],
            "in_frame": True,
        }],
        "frame_idx": cur_idx,
        "stem": stem_cur
    })

    # Build a mapping of final_id -> list of boxes in current frame
    finalid_to_boxes_cur: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    for t in raw_tracks_cur:
        x1, y1, x2, y2, raw_tid, conf, cid = t
        final_tid = resolve_tid(int(raw_tid))
        if final_tid is None:
            continue
        finalid_to_boxes_cur.setdefault(int(final_tid), []).append(
            (int(x1), int(y1), int(x2), int(y2), int(cid))
        )

    # For each candidate, create a single-candidate overlay image
    for i, cand in enumerate(cand_rows):
        cand_final, cand_raw, delta_t, last_fidx, _cname_tail = cand
        cand_final = int(cand_final); cand_raw = int(cand_raw)
        delta_t = float(delta_t); last_fidx = int(last_fidx)
        cname, cbgr = color_assign.get(cand_final, GPT_CANDIDATE_COLORS[min(i, len(GPT_CANDIDATE_COLORS)-1)])

        def _build_single_overlay_from_boxes(base_img, boxes, frame_idx, stem):
            if not boxes:
                return None
            boxes_u = _unique_boxes(boxes)
            main_box, extra_boxes = _main_and_extras(boxes_u)
            ccid = main_box[4] if main_box else boxes_u[0][4]
            canvas = base_img.copy()
            for b in sorted(boxes_u, key=_area_xyxy, reverse=True):
                draw_id_box(canvas, b[:4], cand_final, color_override=(cname, cbgr), draw_tid=True)

            return canvas, {
                "role": "candidate",
                "alias": f"cand#{i}",
                "tid": int(cand_final),
                "color": cname,
                "class_name": classes[ccid] if 0 <= ccid < len(classes) else f"cls{ccid}",
                "delta_t": float(delta_t),
                "bbox": list(map(int, main_box[:4])),
                "extra_bboxes": [list(map(int, b[:4])) for b in (extra_boxes or [])],
                "in_frame": True,
            }, frame_idx, stem

        # 1) If candidate present in current frame use it
        boxes_cur = finalid_to_boxes_cur.get(cand_final, [])
        if boxes_cur:
            result = _build_single_overlay_from_boxes(img_cur, boxes_cur, cur_idx, stem_cur)
            if result is not None:
                canvas, participant, fidx, stm = result
                out_path = save_dir / _fname_cand(stem_cur, new_tid, cand_final, i, "cur", cur_idx, delta_t)
                cv2.imwrite(str(out_path), canvas)
                overlays.append({"path": str(out_path), "participants": [participant], "frame_idx": fidx, "stem": stm})
                continue

        # 2) Otherwise use the last seen frame for that candidate
        stem_prev, img_prev, raw_prev = _load_frame_and_tracks(frame_files, stem2txt, last_fidx)
        if img_prev is None or raw_prev is None:
            continue

        finalid_to_boxes_prev: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        for t in raw_prev:
            x1, y1, x2, y2, raw_tid, conf, cid = t
            final_tid = resolve_tid(int(raw_tid))
            if final_tid is None:
                continue
            finalid_to_boxes_prev.setdefault(int(final_tid), []).append(
                (int(x1), int(y1), int(x2), int(y2), int(cid))
            )

        boxes_prev = finalid_to_boxes_prev.get(cand_final, [])
        result = _build_single_overlay_from_boxes(img_prev, boxes_prev, last_fidx, stem_prev)
        if result is None:
            continue
        canvas, participant, fidx, stm = result
        out_path = save_dir / _fname_cand(stem_prev, new_tid, cand_final, i, "ctx", last_fidx, delta_t)
        cv2.imwrite(str(out_path), canvas)
        overlays.append({"path": str(out_path), "participants": [participant], "frame_idx": fidx, "stem": stm})

    return overlays

# =============================================================================
# Tail store (recent appearances)
# =============================================================================

class TailStore:
    def __init__(self, fps: float, window_sec: float = 300.0):
        self.fps = fps
        self.window = window_sec
        self.store: Dict[int, Tuple[int, float, str]] = {}

    def update(self, track_id: int, frame_idx: int, color_name: str):
        t_sec = frame_idx / self.fps
        self.store[track_id] = (frame_idx, t_sec, color_name)

    def recent_candidates(self, now_frame_idx: int, exclude_tid: int, only_past: bool = True):
        now_t = now_frame_idx / self.fps
        result = []
        for track_id, (frame_idx, t_sec, color_name) in self.store.items():
            if track_id == exclude_tid:
                continue
            if only_past and frame_idx >= now_frame_idx:
                continue
            delta_t = max(0.0, now_t - t_sec)
            if delta_t <= self.window:
                result.append((track_id, delta_t, frame_idx, color_name))
        result.sort(key=lambda x: x[1])
        return result

# =============================================================================
# Remap / resolve and class/color memory
# =============================================================================

remap: Dict[int, Optional[int]] = {}
seen_raw_ids: set = set()
rawID_to_classID: Dict[int, int] = {}
id_colors: Dict[int, str] = {}


def resolve_tid(tid: int):
    """Follow remap chain to the final ID. Return None if mapped to deletion."""
    seen_chain = []
    cur = tid
    while True:
        if cur not in remap:
            final = cur
            break
        seen_chain.append(cur)
        nxt = remap[cur]
        if nxt is None:
            final = None
            break
        if nxt == cur:
            final = cur
            break
        cur = nxt
    for s in seen_chain:
        remap[s] = final
    return final


def parse_target_to_tid(target_id_value, candidate_tid_list):
    if target_id_value is None:
        return None
    if isinstance(target_id_value, int):
        return target_id_value
    if isinstance(target_id_value, str):
        m = re.match(r"^cand#(\d+)$", target_id_value.strip())
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(candidate_tid_list):
                return int(candidate_tid_list[idx])
            return None
        if target_id_value.isdigit():
            return int(target_id_value)
    return None

# =============================================================================
# I/O helpers
# =============================================================================

def write_final_txt(save_dir: Path, stem: str, final_tracks_for_frame: List[Tuple]):
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"frame_{stem}.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for trk in final_tracks_for_frame:
            x1, y1, x2, y2, tid, conf, cid, *rest = trk
            raw_tid = int(rest[0]) if rest else int(tid)
            line = (
                f"{int(tid)} {int(cid)} {"{:.2f}".format(conf)} "
                f"{"{:.1f}".format(x1)} {"{:.1f}".format(y1)} "
                f"{"{:.1f}".format(x2)} {"{:.1f}".format(y2)} "
                f"{raw_tid}"
            )
            f.write(line + "\n")
    return out_path

# =============================================================================
# Main processing loop
# =============================================================================

def main():
    frames_dir = Path(FRAMES_DIR)
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")], key=lambda p: p.name)
    if not frame_files:
        raise SystemExit(f"No frames found in {frames_dir}")

    track_roots = [Path(x) for x in TRACKS_ROOTS]
    stem2txt = build_stem_to_txt_map_multi(track_roots)
    if not stem2txt:
        raise SystemExit(f"No txt files found under any of: {', '.join(str(r) for r in track_roots)}")

    dir_overlay = out_root / OVERLAY_SUBDIR
    dir_gpt = out_root / GPT_OVERLAY_SUBDIR
    dir_final_txt = out_root / FINAL_TXT_SUBDIR
    decisions_path = out_root / "decisions.jsonl"

    tail_store = TailStore(FPS, WINDOW_SEC)

    for idx, fimg in enumerate(frame_files):
        stem = normalize_stem(fimg.stem)
        txt_path = stem2txt.get(stem)
        if txt_path is None:
            continue

        img = cv2.imread(str(fimg))
        if img is None:
            print(f"[WARN] cannot read frame image: {fimg}")
            continue

        raw_tracks = read_tracks_txt(txt_path)
        if not raw_tracks:
            continue

        current_raw_ids = {int(t[4]) for t in raw_tracks}

        if idx < WARMUP_FRAMES:
            for tid in current_raw_ids:
                remap.setdefault(tid, tid)
            new_raw_ids = []
        else:
            new_raw_ids = [tid for tid in current_raw_ids if tid not in seen_raw_ids]

        for new_tid in new_raw_ids:
            if new_tid in remap:
                continue

            cid_list = [int(t[6]) for t in raw_tracks if int(t[4]) == new_tid]
            cid = cid_list[0] if cid_list else 0
            class_name = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"cls{cid}"
            print(f"[INFO {idx}] New ID {new_tid}: {class_name}")

            cand_rows_all = tail_store.recent_candidates(idx, exclude_tid=new_tid, only_past=True)
            cand_final_map: Dict[int, Dict] = {}
            for (raw_id, delta_t, last_fidx, color_name) in cand_rows_all:
                final_id = resolve_tid(int(raw_id))
                if final_id is None:
                    continue
                prev = cand_final_map.get(final_id)
                if prev is None or delta_t < prev["delta_t"]:
                    cand_final_map[final_id] = {"raw_id": int(raw_id), "delta_t": float(delta_t), "last_fidx": int(last_fidx), "color_name": color_name}

            cand_final_list = [(int(final_id), data["raw_id"], data["delta_t"], data["last_fidx"], data["color_name"]) for final_id, data in cand_final_map.items()]
            cand_final_list.sort(key=lambda x: x[2])
            cand_final_filtered = [c for c in cand_final_list if rawID_to_classID.get(c[0]) == cid] or cand_final_list
            cand_selected = cand_final_filtered[:MAX_CANDIDATES]
            candidate_tid_list = [c[0] for c in cand_selected]

            new_box = next(((t[0], t[1], t[2], t[3]) for t in raw_tracks if int(t[4]) == new_tid), None)
            if new_box is None:
                remap[new_tid] = new_tid
                continue

            overlays = build_gpt_overlays_multi(
                cur_idx=idx,
                frame_files=frame_files,
                stem2txt=stem2txt,
                img_cur=img, stem_cur=stem,
                new_tid=new_tid, new_cid=cid, new_box=new_box,
                raw_tracks_cur=raw_tracks,
                cand_rows=cand_selected,
                classes=CLASSES, save_dir=dir_gpt, max_cands=MAX_CANDIDATES
            )

            cand_color_by_tid = {}
            for o in overlays[1:]:
                if o.get("participants"):
                    p = o["participants"][0]
                    cand_color_by_tid[int(p["tid"])] = p.get("color")

            candidates = []
            for i, cand in enumerate(cand_selected):
                final_tid, raw_tid, delta_t, last_fidx, _ = cand
                candidates.append(CandidateTrack(tid=int(final_tid), delta_t=float(delta_t), color_name=cand_color_by_tid.get(int(final_tid))))

            H, W = img.shape[:2]
            event = NewIdEvent(
                current_idx=idx,
                class_name=class_name,
                frame_width=W,
                frame_height=H,
                fps=FPS,
                new_id=str(new_tid),
                current_overlay_path=str(overlays[0]["path"]) if overlays else None,
                participants=overlays[0]["participants"] if overlays else [],
                candidates=candidates,
                past_overlays=[{"path": o["path"], "participants": o["participants"], "frame_idx": o["frame_idx"], "stem": o["stem"]} for o in overlays[1:]] if len(overlays) > 1 else []
            )

            decision = adjudicate_new_id(event)

            action = decision["new_id_decision"].get("action")
            target = decision["new_id_decision"].get("target_id")

            with decisions_path.open("a", encoding="utf-8") as fp_out:
                fp_out.write(json.dumps(decision, ensure_ascii=False) + "\n")

            print(f"[GPT {idx}] : {decision}")
            if action == "assign":
                mapped_tid = parse_target_to_tid(target, candidate_tid_list)
                if mapped_tid is None:
                    raise ValueError(f"Invalid target_id: {target} for new_tid={new_tid}")
                else:
                    target_final = resolve_tid(int(mapped_tid))
                    if target_final is None:
                        remap[new_tid] = None
                        print(f"[WARN] target {mapped_tid} deleted")
                    else:
                        remap[new_tid] = int(target_final)
                        print(f"[ASSIGN] {new_tid} → {remap[new_tid]}")
            elif action == "new":
                remap[new_tid] = new_tid
                print(f"[NEW] keep {new_tid}")
            elif action == "delete":
                remap[new_tid] = None
                print(f"[DELETE] drop {new_tid}")
            else:
                raise ValueError(f"Unknown action: {action}")

        seen_raw_ids.update(current_raw_ids)

        best_by_final_tid: Dict[int, Tuple] = {}
        for (x1, y1, x2, y2, tid, conf, cid) in raw_tracks:
            final_tid = resolve_tid(int(tid))
            if final_tid is None:
                continue
            cand = (x1, y1, x2, y2, final_tid, conf, cid, int(tid))
            prev = best_by_final_tid.get(final_tid)
            if prev is None:
                best_by_final_tid[final_tid] = cand
            else:
                prev_conf = float(prev[5])
                if (conf > prev_conf) or (conf == prev_conf and _area_xyxy((x1, y1, x2, y2)) > _area_xyxy(prev[:4])):
                    best_by_final_tid[final_tid] = cand

        final_tracks_for_frame: List[Tuple] = list(best_by_final_tid.values())

        if SAVE_FRAME_OVERLAY:
            overlay_path = out_root / OVERLAY_SUBDIR / f"{stem}.jpg"
            save_frame_overlay(img, final_tracks_for_frame, CLASSES, overlay_path)

        _ = write_final_txt(out_root / FINAL_TXT_SUBDIR, stem, final_tracks_for_frame)

        for trk in final_tracks_for_frame:
            x1, y1, x2, y2, final_tid, conf, cid, *rest = trk
            raw_tid = int(rest[0]) if rest else int(final_tid)
            cname, _ = color_for_id(int(final_tid))
            if int(final_tid) != int(raw_tid):
                tail_store.update(int(final_tid), idx, cname)
                tail_store.store.pop(int(raw_tid), None)
            else:
                tail_store.update(int(raw_tid), idx, cname)
            rawID_to_classID[int(final_tid)] = int(cid)
            rawID_to_classID[int(raw_tid)] = int(cid)
            id_colors[int(final_tid)] = cname

    with (out_root / "id_colors.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(id_colors.items())}, f, ensure_ascii=False, indent=2)

    print(f"Done. Decisions saved → {decisions_path}")
    print(f"Final TXT saved under → {out_root / FINAL_TXT_SUBDIR}")
    if SAVE_FRAME_OVERLAY:
        print(f"Frame overlays saved under → {out_root / OVERLAY_SUBDIR}")
    print(f"ID color map saved → {out_root / 'id_colors.json'}")


if __name__ == "__main__":
    main()
