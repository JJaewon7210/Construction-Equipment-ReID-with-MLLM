"""Tracking pipeline with optional video output.

It creates:
* A results directory tracked_results_partXXX   (TXT per frame)
* A video clip  tracked_partXXX.mp4  (visualized tracks)
"""
import os
import argparse
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from boxmot import BoostTrack, BotSort, StrongSort, DeepOcSort
import random

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Utility
def parse_detections(txt_path: str) -> np.ndarray:
    """
    Parse a detection text file and return a float32 array of shape [N,6]
    with columns [x1, y1, x2, y2, conf, cls].

    Expected input format per line: 'cls x1 y1 x2 y2 conf' (whitespace-separated).
    This function reads the entire file as ASCII and converts to a numpy array
    which is fast for large files.
    """
    if not os.path.exists(txt_path):
        return np.empty((0, 6), dtype=np.float32)

    with open(txt_path, "rb") as f:
        buf = f.read().strip()
    if not buf:
        return np.empty((0, 6), dtype=np.float32)

    arr = np.fromstring(buf.decode("ascii"), sep=" ", dtype=np.float32)
    if arr.size < 6:
        return np.empty((0, 6), dtype=np.float32)

    n = (arr.size // 6) * 6
    arr = arr[:n].reshape(-1, 6)  # [cls, x1, y1, x2, y2, conf]
    # reorder to [x1, y1, x2, y2, conf, cls]
    arr = arr[:, [1, 2, 3, 4, 5, 0]]
    return arr


def process_one_partition(
    partition_idx: int,
    raw_tracks_by_stem: dict,
    frame_info_list: list,
    args,
):
    """Perform second-pass processing and write outputs for a single partition."""
    start_time = time.time()
    output_root = Path(args.output_root)
    txt_dir = output_root / f"tracked_results_part{partition_idx:03d}"
    txt_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_root / f"tracked_part{partition_idx:03d}.mp4"

    # Initialize video writer lazily after we know frame size
    writer = None
    if args.make_video and frame_info_list:
        first_img = cv2.imread(frame_info_list[0]["path"])
        if first_img is not None:
            h, w = first_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

    # Iterate through frames in the partition and write per-frame TXT and video frames
    for info in frame_info_list:
        stem, fpath = info["stem"], info["path"]
        img = cv2.imread(fpath)
        if img is None:
            continue

        tracks = raw_tracks_by_stem.get(stem, np.empty((0, 7)))

        # Save per-frame TXT with format: id class score x1 y1 x2 y2
        with (txt_dir / f"{stem}.txt").open("w") as fp:
            for t in tracks:
                x1, y1, x2, y2, tid, conf, cid = t[:7]
                fp.write(f"{int(tid)} {int(cid)} {conf:.2f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        # Draw boxes and labels and write to video if requested
        if writer is not None:
            vis = img.copy()
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                conf = t[5]
                tid, cid = int(t[4]), int(t[6])
                cls_name = args.classes[cid] if cid < len(args.classes) else f"cls{cid}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{cls_name}-{tid} {conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            writer.write(vis)

    total_time = time.time() - start_time
    print(f"[PARTITION {partition_idx:03d}] Processed {len(frame_info_list)} frames in {total_time:.2f}s")

    if writer is not None:
        writer.release()
        print(f"[PARTITION {partition_idx:03d}] Video saved -> {video_path}")
    print(f"[PARTITION {partition_idx:03d}] TXT saved -> {txt_dir}")


def main():
    parser = argparse.ArgumentParser(description="partition-based post-processing tracker")

    # Paths and options
    parser.add_argument("--tracker", type=str, default="boosttrack",
                        help="Tracker to use: boosttrack|botsort|strongsort|deepocsort")
    parser.add_argument("--frames", type=str, default="data/Busan_01/images",
                        help="Directory with extracted image frames")
    parser.add_argument("--dets", type=str, default="detections/Busan_01",
                        help="Directory with detection txt files")
    parser.add_argument("--reid_weights", type=str, default="weights/osnet_x0_75_YOUTUBE.pt",
                        help="Path to ReID weights")
    parser.add_argument("--output_root", default="tracked_outputs/Busan_01/tracked_boosttrack+ReID(Youtube)",
                        help="Root directory for outputs (videos + txt)")

    # Classes & video options
    parser.add_argument("--classes", nargs="+", default=["truck", "excavator"],
                        help="List of class names (index == class_id)")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for output videos")
    parser.add_argument("--make_video", default=False, help="Write a video clip per partition if set")

    # Tracking / partitioning parameters
    parser.add_argument("--partition_size", type=int, default=1800, help="Frames per partition")
    
    args = parser.parse_args()
    
    # Paths and device
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate tracker
    if args.tracker == "boosttrack":
        tracker = BoostTrack(
            reid_weights=Path(args.reid_weights),
            device=device,
            max_age=30 * 30,
            min_hits=15,
            det_thresh=0.25,
            iou_threshold=0.3,
            min_box_area=0,
            aspect_ratio_thresh=3.0,
            lambda_iou=0.3,
            lambda_mhd=0.25,
            lambda_shape=0.1,
            use_ecc=False,
            s_sim_corr=True,
            use_rich_s=True,
            use_sb=True,
            use_vt=True,
            half=True,
            with_reid=True,
        )
    elif args.tracker == "botsort":
        tracker = BotSort(
            reid_weights=Path(args.reid_weights),
            device=device,
            half=True,
            track_low_thresh=0.25,
            new_track_thresh=0.25,
            track_high_thresh=0.50,
            track_buffer=30 * 30,
            frame_rate=30,
            with_reid=True,
            per_class=False,
            fuse_first_associate=True,
        )
    elif args.tracker == "strongsort":
        tracker = StrongSort(
            reid_weights=Path(args.reid_weights),
            device=device,
            half=True,
            per_class=False,
            min_conf=0.25,
            max_age=30 * 30,
            n_init=15,
        )
    elif args.tracker == "deepocsort":
        tracker = DeepOcSort(
            reid_weights=Path(args.reid_weights),
            device=device,
            half=True,
            max_age=30 * 30,
            min_hits=15,
            det_thresh=0.25,
            iou_threshold=0.3,
            alpha_fixed_emb=1.0,
            cmc_off=True,
        )
    else:
        raise ValueError(f"Unknown tracker: {args.tracker}")

    # Enumerate image frames
    frame_files = sorted([
        f for f in os.listdir(args.frames)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # Main loop: feed frames to the tracker and flush outputs every partition_size frames
    partition_idx = 0
    frames_in_partition = 0
    txt_dir = None
    writer = None
    t0 = time.time()

    for gi, fname in enumerate(frame_files):
        fpath = os.path.join(args.frames, fname)
        stem = os.path.splitext(fname)[0]
        img = cv2.imread(fpath)

        # Prepare partition output directories/writer at the beginning of each partition
        if frames_in_partition == 0:
            txt_dir = output_root / f"tracked_results_part{partition_idx:03d}"
            txt_dir.mkdir(parents=True, exist_ok=True)

            if args.make_video:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vpath = output_root / f"tracked_part{partition_idx:03d}.mp4"
                writer = cv2.VideoWriter(str(vpath), fourcc, args.fps, (w, h))

        # Parse detections and update tracker
        det_path = os.path.join(args.dets, f"{stem}.txt")
        dets = parse_detections(det_path)  # float32 array [x1,y1,x2,y2,conf,cls]

        tracks = tracker.update(dets, img)

        # Write per-frame TXT: id class score x1 y1 x2 y2
        if tracks.size:
            lines = [
                f"{int(t[4])} {int(t[6])} {t[5]:.2f} {t[0]:.1f} {t[1]:.1f} {t[2]:.1f} {t[3]:.1f}\n"
                for t in tracks
            ]
        else:
            lines = []
        with (txt_dir / f"{stem}.txt").open("w") as fp:
            fp.writelines(lines)

        # Optional immediate video frame writing
        if writer is not None:
            vis = img.copy()
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                tid, cid, conf = int(t[4]), int(t[6]), float(t[5])
                cls_name = args.classes[cid] if cid < len(args.classes) else f"cls{cid}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{cls_name}-{tid} {conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            writer.write(vis)

        frames_in_partition += 1

        # Check end-of-partition or end-of-sequence
        at_partition_end = frames_in_partition >= args.partition_size
        at_last_frame = gi == len(frame_files) - 1
        if at_partition_end or at_last_frame:
            if writer is not None:
                writer.release()
                writer = None
            dt = time.time() - t0
            print(f"[partition {partition_idx:03d}] {frames_in_partition} frames in {dt:.2f}s "
                  f"({1000*dt/max(frames_in_partition,1):.1f} ms/frame)")
            partition_idx += 1
            frames_in_partition = 0
            t0 = time.time()

    print("All done!")

if __name__ == "__main__":
    main()
