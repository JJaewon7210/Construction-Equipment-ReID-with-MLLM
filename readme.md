
# Construction Equipment Re-ID with MLLM

A comprehensive computer vision pipeline for construction equipment detection, tracking, and re-identification using Multi-modal Large Language Models (MLLM). This system combines YOLO object detection, multi-object tracking, and GPT-based adjudication for robust construction equipment monitoring.

## Overview

This project implements an end-to-end pipeline for:
- **Detection**: Construction equipment detection using YOLO11
- **Tracking**: Multi-object tracking with state-of-the-art algorithms
- **Re-identification**: OSNet-based re-identification models
- **Adjudication**: GPT-powered intelligent decision making for track association

## Features

- 🏗️ **Construction Equipment Focus**: Specialized for dump trucks and excavators
- 🎯 **High-Performance Detection**: YOLO11-based detection with custom weights
- 🔄 **Robust Tracking**: Multiple tracking algorithms (BoostTrack, BotSort, StrongSort, DeepOcSort)
- 🧠 **AI-Powered Adjudication**: GPT-based intelligent track association
- 📊 **Comprehensive Validation**: Built-in validation tools for detection and tracking results
- 🎥 **Video Output**: Visualization capabilities for tracking results

## Project Structure

```
Construction-Equipment-ReID-with-MLLM/
├── c001_detect.py          # Object detection pipeline
├── c001_validate.py        # Detection validation
├── c002_track.py           # Multi-object tracking
├── c003_gpt_adjudicator.py # GPT-based track adjudication
├── c003_postprocess.py     # Post-processing pipeline
├── c003_validate.py        # Tracking validation
├── data/                   # Input datasets
├── detections/             # Detection results
├── tracked_outputs/        # Tracking results
├── trackeval_workspace/    # Evaluation workspace
└── weights/                # Model weights
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenAI API access
- BoxMOT (for tracking)
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Construction-Equipment-ReID-with-MLLM
```

2. Install dependencies:
```bash
pip install ultralytics torch opencv-python numpy openai boxmot
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Data Download

### Case Study Dataset
Download the construction equipment images and labels:

- **Busan_01**: https://drive.google.com/file/d/16d8NEIsFLlb3EyOLh9mHtEnOEZt5Ybnc/view?usp=drive_link
- **Busan_02**: https://drive.google.com/file/d/1WmolxsDBFZepX4MA7sjc5duyOir0hFPI/view?usp=drive_link

Place the downloaded data in the `data/` directory.

### Model Weights

#### YOLO11 Detection Weights
- **mocs_acid_y11m (+pump +worker)**: https://drive.google.com/file/d/1JjMTWO1shp9_RDTyWdPTQnWmZti62KIu/view?usp=drive_link
- **mocs_acid_y11s (+pump +worker)**: https://drive.google.com/file/d/1By5iBjVLr4GJcSlQqIRXcD1XLXxaMlbq/view?usp=drive_link

#### Re-ID Model Weights
- **osnet_x0_75_AIHUB.pt**: https://drive.google.com/file/d/1oq8oiOh76lGgswldDwWva2Z4DSPRgFkI/view?usp=drive_link
- **osnet_x0_75_AIHUB+YOUTUBE+VeRI.pt**: https://drive.google.com/file/d/1Bxp50riU59hxSIM6sLfqz4Ik66SSNFtf/view?usp=drive_link
- **osnet_x0_75_VeRI.pt**: https://drive.google.com/file/d/1VJTlVuM_pA-FR2U4XqV44UmpmeKcnwgH/view?usp=drive_link
- **osnet_x0_75_YOUTUBE.pt**: https://drive.google.com/file/d/1dIGHJMnG4wl4kblg3haI1P0NxEfkkRT9/view?usp=drive_link

Place all weights in the `weights/` directory.

## Usage

Follow these steps to run the complete pipeline:

### Step 1: Object Detection
```bash
python c001_detect.py
```
Detects construction equipment in input images using YOLO11.

### Step 2: Validate Detection Results
```bash
python c001_validate.py
```
Validates and analyzes detection performance.

### Step 3: Multi-Object Tracking
```bash
python c002_track.py
```
Performs multi-object tracking on detected equipment.

### Step 4: Post-Processing with GPT Adjudication
```bash
python c003_postprocess.py
```
Applies GPT-based intelligent adjudication for track association and refinement.

### Step 5: Validate Tracking Results
```bash
python c003_validate.py
```
Evaluates tracking performance and generates metrics.

## Data Format

### Ground Truth Labels
Each label file follows the format:
```
{track_ID} {class_ID} {confidence} {x1} {y1} {x2} {y2}
```

Where:
- `track_ID`: Unique identifier for the tracked object
- `class_ID`: 0 for dump_truck, 1 for excavator
- `confidence`: Always 1 for ground truth
- `x1, y1, x2, y2`: Bounding box coordinates in absolute pixels

### Detection Output Format
Detection results are saved as text files with format:
```
{class_ID} {x1} {y1} {x2} {y2} {confidence}
```

## Performance

Our system demonstrates state-of-the-art performance across detection and tracking tasks on construction equipment datasets. The comprehensive evaluation includes YOLO detection performance on ACID and MOCS datasets, and tracking validation on Busan case study scenarios.

### Detection Performance

The YOLO11 models were validated on ACID and MOCS datasets, showing excellent precision and recall for construction equipment detection:

#### YOLO11m Detection Results
| Metric | Value | Description |
|--------|--------|-------------|
| **Training Time** | 69,250.2s | Total training duration |
| **Precision** | 0.919 | Detection precision |
| **Recall** | 0.852 | Detection recall |
| **mAP50** | 0.912 | Mean Average Precision at IoU=0.5 |
| **mAP50-95** | 0.811 | Mean Average Precision at IoU=0.5:0.95 |

#### YOLO11s Detection Results
| Metric | Value | Description |
|--------|--------|-------------|
| **Training Time** | 42,988.2s | Total training duration |
| **Precision** | 0.927 | Detection precision |
| **Recall** | 0.821 | Detection recall |
| **mAP50** | 0.901 | Mean Average Precision at IoU=0.5 |
| **mAP50-95** | 0.792 | Mean Average Precision at IoU=0.5:0.95 |

### Tracking Performance

Tracking validation was conducted on Busan_01 (Scene A) and Busan_02 (Scene B) case study datasets. The results demonstrate significant improvements when incorporating EquipReID and MLLM components.

#### Method Comparison (Micro-averaged Evaluation)

| Scene | Method | IDP | IDR | IDF1 | IDSW | MOTA | HOTA |
|-------|--------|-----|-----|------|------|------|------|
| A | BoostTrack | 0.750 | 0.561 | 0.642 | 30 | 0.681 | 0.654 |
| A | +EquipReID | 0.748 | 0.561 | 0.641 | 26 | 0.683 | 0.655 |
| A | **+EquipReID +MLLM** | **0.930** | **0.678** | **0.784** | **3** | **0.704** | **0.725** |
| B | BoostTrack | 0.546 | 0.512 | 0.528 | 32 | 0.898 | 0.558 |
| B | +EquipReID | 0.662 | 0.621 | 0.641 | 24 | 0.898 | 0.637 |
| B | **+EquipReID +MLLM** | **0.871** | **0.785** | **0.825** | **2** | **0.871** | **0.748** |

#### Tracker and ReID Dataset Comparison

| Scene | Tracker | ReID Dataset | IDP | IDR | IDF1 | IDSW | HOTA |
|-------|---------|-------------|-----|-----|------|------|------|
| A | BoostTrack | AI-Hub | 0.772 | 0.579 | 0.662 | 27 | 0.674 |
| A | BoostTrack | YouTube | 0.793 | 0.595 | 0.680 | 26 | 0.681 |
| A | BoostTrack | VeRI | 0.769 | 0.577 | 0.659 | 25 | 0.662 |
| A | BoostTrack | **EquipReID (Combined)** | 0.748 | 0.561 | 0.641 | 26 | 0.655 |
| A | StrongSort | AI-Hub | 0.782 | 0.632 | 0.699 | 40 | 0.673 |
| A | StrongSort | YouTube | 0.733 | 0.589 | 0.653 | 47 | 0.649 |
| A | StrongSort | VeRI | 0.720 | 0.576 | 0.640 | 48 | 0.643 |
| A | StrongSort | **EquipReID (Combined)** | 0.829 | 0.672 | 0.742 | 47 | 0.710 |
| B | BoostTrack | AI-Hub | 0.651 | 0.611 | 0.630 | 26 | 0.627 |
| B | BoostTrack | YouTube | 0.617 | 0.579 | 0.597 | 29 | 0.594 |
| B | BoostTrack | VeRI | 0.638 | 0.598 | 0.617 | 27 | 0.615 |
| B | BoostTrack | **EquipReID (Combined)** | 0.662 | 0.621 | 0.641 | 24 | 0.637 |
| B | StrongSort | AI-Hub | 0.563 | 0.552 | 0.557 | 63 | 0.601 |
| B | StrongSort | YouTube | 0.555 | 0.544 | 0.550 | 56 | 0.595 |
| B | StrongSort | VeRI | 0.571 | 0.560 | 0.566 | 53 | 0.601 |
| B | StrongSort | **EquipReID (Combined)** | 0.564 | 0.554 | 0.559 | 54 | 0.602 |

## Citation

If you use this work in your research, please cite:
```bibtex
Multiple Object Tracking and Re-Identification of Construction Equipment using Multimodal Large Language Model
```

## Contact

jjw0127@yonsei.ac.kr
