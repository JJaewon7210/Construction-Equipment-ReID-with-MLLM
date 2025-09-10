from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from typing import List, Dict, Optional

def create_image_list_files(frames_folder: Path, partition_size: int = 3000) -> List[Path]:
    """
    Recursively search for images under frames_folder and split them into
    text files containing up to partition_size image paths each.

    Returns a list of Path objects pointing to the created text files.
    """
    frames_folder = Path(frames_folder)
    image_files: List[Path] = []
    # recursively search for common image extensions
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(frames_folder.rglob(ext))
    image_files = sorted(image_files)

    txt_files: List[Path] = []
    for i in range(0, len(image_files), partition_size):
        partition = image_files[i : i + partition_size]
        idx = i // partition_size
        txt_filename = f"image_list_temp_{idx:02d}.txt"
        txt_path = Path(txt_filename)

        with txt_path.open("w", encoding="utf-8") as f:
            for img_path in partition:
                # write absolute/resolved path to avoid ambiguity
                f.write(f"{str(img_path.resolve())}\n")

        txt_files.append(txt_path)
        print(f"Created {txt_filename} with {len(partition)} images")

    return txt_files


def filter_yolo_annotations(folder_path: Path, class_ids: List[str] = ["0", "1"]) -> None:
    """
    In all .txt files in folder_path, keep only the lines whose first token (class id)
    is in class_ids. Modifies files in-place.
    """
    txt_files = list(Path(folder_path).glob("*.txt"))
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        filtered_lines = [line for line in lines if line.strip() and line.strip().split()[0] in class_ids]
        with open(txt_file, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)


def iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute IoU of two bounding boxes in the format:
    [x1, y1, x2, y2, score, label]. Only x1..y2 are used for IoU.

    If labels differ, IoU is considered 0 (for matching by label).
    """
    x1, y1, x2, y2, _, lbl = bbox1
    x1p, y1p, x2p, y2p, _, lblp = bbox2

    # If labels differ, treat as non-overlapping for matching purposes
    if lbl != lblp:
        return 0.0

    inter_x1 = max(x1, x1p)
    inter_y1 = max(y1, y1p)
    inter_x2 = min(x2, x2p)
    inter_y2 = min(y2, y2p)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    area2 = max(0.0, (x2p - x1p)) * max(0.0, (y2p - y1p))
    union_area = area1 + area2 - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area


def get_color(label_id: int):
    """
    Return a deterministic BGR color tuple for a given class id.
    """
    colors = [
        (255, 0, 0),    # blue-ish (BGR)
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 255, 0),  # yellow/cyan mix
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
    ]
    return colors[label_id % len(colors)]


def save_yolo_predictions(
    model_path: Path,
    val_list_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.25,
    device: str = "cuda:0",
    save_images: bool = False,
    class_mapping: Optional[Dict[int, int]] = None,
    augment: bool = False,
) -> None:
    """
    Run the Ultralytics YOLO model on images listed in val_list_path.

    - Collect detections in the format [x1, y1, x2, y2, score, label_id]
    - Save a .txt per image with lines: label x1 y1 x2 y2 score
    - If save_images True, draw predicted boxes and (if available) true
      label boxes and save images under output_dir/images.
    """
    model_path, val_list_path, output_dir = Path(model_path), Path(val_list_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_output_dir = None
    if save_images:
        img_output_dir = output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    detection_model = YOLO(str(model_path))
    # Try to move model to the desired device if the method exists
    try:
        detection_model.to(device)
    except Exception:
        # Some versions may not support .to() or device strings; ignore if it fails
        pass

    # Read image paths
    with val_list_path.open("r", encoding="utf-8") as f:
        img_paths_list = [line.strip() for line in f if line.strip()]

    # (1) Collect detections for every image as [x1,y1,x2,y2,score,label_id]
    detections: Dict[int, List[List[float]]] = {}
    for i, img_path_str in enumerate(img_paths_list):
        img_path = Path(img_path_str)
        # Use the model predict API; pass conf and device if supported
        try:
            results = detection_model.predict(source=str(img_path), verbose=False, augment=augment, conf=confidence_threshold, device=device)
        except TypeError:
            # Fallback if keyword args aren't supported by this version
            results = detection_model.predict(str(img_path), verbose=False, augment=augment)

        bboxes: List[List[float]] = []
        for r in results:
            # r.boxes.data is expected to be iterable of [x1,y1,x2,y2,score,label]
            for box in r.boxes.data:
                x1, y1, x2, y2, s, lid = box
                if float(s) >= confidence_threshold:
                    bboxes.append([float(x1), float(y1), float(x2), float(y2), float(s), int(lid)])
        detections[i] = bboxes

    # (2) Save results to .txt files and optionally save images
    for i, img_path_str in enumerate(img_paths_list):
        img_path = Path(img_path_str)
        out_txt = output_dir / f"{img_path.stem}.txt"

        # Save prediction text file: label x1 y1 x2 y2 score
        with out_txt.open("w", encoding="utf-8") as f:
            for bbox in detections.get(i, []):
                x1, y1, x2, y2, score, label_id = bbox
                f.write(f"{label_id} {x1} {y1} {x2} {y2} {score}\n")

        if save_images and img_output_dir is not None:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: could not read image {img_path}")
                continue

            img_pred = img.copy()
            img_true = img.copy()

            # Draw predicted boxes
            for bbox in detections.get(i, []):
                x1, y1, x2, y2, score, label_id = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                display_label = int(label_id)
                if class_mapping is not None:
                    display_label = class_mapping.get(display_label, display_label)
                color = get_color(display_label)

                cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                label_text = f"{display_label:2d}: {score:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img_pred, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                cv2.putText(img_pred, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save predicted image
            output_img_path = img_output_dir / f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(output_img_path), img_pred)

            # Draw true labels if available
            true_label_path = _get_label_path_from_image(img_path_str)

            if true_label_path.exists():
                img_height, img_width = img.shape[:2]
                true_bboxes: List[List[float]] = []
                with true_label_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        # YOLO label format: class_id, cx, cy, w, h (normalized)
                        if len(parts) >= 5:
                            label_id = int(parts[0])
                            center_x = float(parts[1]) * img_width
                            center_y = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            x1 = int(center_x - width / 2)
                            y1 = int(center_y - height / 2)
                            x2 = int(center_x + width / 2)
                            y2 = int(center_y + height / 2)
                            score = float(parts[5]) if len(parts) > 5 else 1.0
                            true_bboxes.append([x1, y1, x2, y2, score, label_id])

                for bbox in true_bboxes:
                    x1, y1, x2, y2, score, label_id = bbox
                    color = get_color(label_id)
                    cv2.rectangle(img_true, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label_id:2d}"
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(img_true, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(img_true, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                output_true_img_path = img_output_dir / f"{img_path.stem}_true.jpg"
                cv2.imwrite(str(output_true_img_path), img_true)


def _get_label_path_from_image(img_path_str: str) -> Path:
    p = Path(img_path_str)
    parts = list(p.parts)
    # Replace the first occurrence of "images" (case-insensitive) with "labels"
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix('.txt')

    # Fallback: try finding a sibling 'labels' folder at ancestor levels
    for anc in p.parents:
        try:
            rel = p.relative_to(anc)
        except Exception:
            continue
        candidate = anc / 'labels' / rel.with_suffix('.txt')
        if candidate.exists():
            return candidate

    # Final fallback: same directory, change extension to .txt
    return p.with_suffix('.txt')


if __name__ == "__main__":
    yolo_model_path = Path("weights/mocs_acid_y11m (+pump +worker)/weights/best.pt")
    output_dir = Path("detections/Busan_01/")
    frames_dir = Path("data/Busan_01/images")
    class_ids = ["0", "1"]  # 0 : dump truck, 1 : excavator

    print("Creating temporary image list files...")
    txt_frame_list_files = create_image_list_files(frames_dir, partition_size=1800)

    for i, txt_file in enumerate(txt_frame_list_files):
        print(f"Processing {txt_file}...")
        with txt_file.open('r', encoding='utf-8') as f:
            num_frames = sum(1 for _ in f)
        t0 = time.time()
        save_yolo_predictions(
            yolo_model_path,
            txt_file,
            output_dir,
            confidence_threshold=0.25,
            device="cuda:0",
            save_images=False,
            augment=False,
        )
        dt = time.time() - t0
        print(f"[partition {i:03d}] {num_frames} frames in {dt:.2f}s ({1000*dt/max(num_frames,1):.1f} ms/frame)")
        print(f"Completed partition {i+1}/{len(txt_frame_list_files)}")

    # Remove temporary txt files
    for txt_file in txt_frame_list_files:
        if txt_file.exists():
            txt_file.unlink()
    print("Deleted temporary text files")

    # Filter and convert labels
    filter_yolo_annotations(output_dir, class_ids)
