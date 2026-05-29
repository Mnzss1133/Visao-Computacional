# Visao-Computacional

# Weapon Detection and Behavioral Risk Classification in Surveillance Using Deep Learning

> Published at the **XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026)**  
> September 29th to October 2nd, 2026 — Salvador, BA

**Authors:** Gabriel Mendes, João Pedro Alencar, João Pedro Menezes, Rigel P. Fernandes, Cassius Figueredo, Thiago S. de Souza

---

## Overview

This project presents a deep learning-based pipeline for **suspicious activity detection in surveillance systems**. The system integrates weapon detection, human pose estimation, and rule-based behavioral analysis to classify each detected person into one of four risk levels in real time.

```
Input Frame
    │
    ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Person Detection   │     │   Weapon Detection   │
│  YOLOv8n (COCO)     │     │  YOLOv8s (fine-tuned)│
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
         ▼                             ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Pose Estimation    │     │   Fallback Detector  │
│  YOLOv8n-pose       │     │  COCO (knife, bat..) │
│  17 COCO keypoints  │     └──────────┬───────────┘
└────────┬────────────┘                │
         │                             │
         └──────────────┬──────────────┘
                        ▼
           ┌────────────────────────┐
           │   BehaviorAnalyzer     │
           │  - Hand elevation      │
           │  - Knee flexion        │
           │  - Trunk inclination   │
           │  - Arm extension       │
           │  - Weapon proximity    │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │   Risk Classification  │
           │  🟢 Normal             │
           │  🟡 Suspicious         │
           │  🟠 High Risk          │
           │  🔴 Critical           │
           └────────────────────────┘
```

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | **0.953** |
| mAP@50-95 | **0.725** |
| Precision | **0.972** |
| Recall | **0.912** |

> Model trained for 45 epochs on the [Gun Detection Dataset](https://universe.roboflow.com/workspace-1qko2/gun-detection-ghlzd) using YOLOv8s on Google Colab (GPU 25GB VRAM).

---

## Risk Classification Logic

| Condition | Risk Level | Score |
|-----------|-----------|-------|
| Weapon detected + arm extended | 🔴 Critical | 10 |
| Weapon detected + trunk inclined | 🟠 High Risk | 5 |
| Weapon detected only | 🟠 High Risk | 5 |
| Weapon detected + hand raised | 🟡 Suspicious | 3 |
| Raised hand or extended arm (no weapon) | 🟡 Suspicious | 2 pts each |
| Crouching or inclination (no weapon) | 🟡 Suspicious | 2 pts each |

Weapon-to-hand proximity is verified by computing the Euclidean distance between the weapon bounding box center and wrist keypoints (COCO indices 9 and 10). Distance < 100px → person considered armed.

---

## Pose Keypoints (COCO Format)

```
0: nose          5: left shoulder    10: right wrist
1: left eye      6: right shoulder   11: left hip
2: right eye     7: left elbow       12: right hip
3: left ear      8: right elbow      13: left knee
4: right ear     9: left wrist       14: right knee
                                     15: left ankle
                                     16: right ankle
```

Confidence threshold: **0.3** (keypoints below this are discarded).

---

## Project Structure

```
├── data/
│   └── data.yaml                  # Dataset config (train/val/test paths)
├── models/
│   └── best_weapon.pt             # Fine-tuned YOLOv8s weapon detector
├── src/
│   ├── weapon_detector.py         # WeaponDetectorFallback class
│   ├── pose_estimator.py          # YOLOv8n-pose wrapper
│   ├── behavior_analyzer.py       # BehaviorAnalyzer class
│   └── pipeline.py                # Full detection pipeline
├── notebooks/
│   └── train_weapon_detector.ipynb  # Training notebook (Google Colab)
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
├── pics/
│   └── deteccao_arma.png          # Sample detection output
└── README.md
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Mnzss1133/Visao-Computacional
cd Visao-Computacional

# Install dependencies
pip install ultralytics opencv-python matplotlib
```

---

## Training

The weapon detection model was fine-tuned from YOLOv8s pretrained weights:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data/data.yaml",
    epochs=45,
    imgsz=640,
    batch=16,
    patience=10,
    augment=True,
    mosaic=0.3,
    lr0=0.01,
    weight_decay=0.0005,
    dropout=0.1,
    project="runs/weapon_detector",
    name="yolov8s_45ep"
)
```

Dataset: [Gun Detection v1 — Roboflow Universe](https://universe.roboflow.com/workspace-1qko2/gun-detection-ghlzd)

---

## Running the Pipeline

```python
from src.pipeline import SurveillancePipeline

pipeline = SurveillancePipeline(
    person_model="yolov8n.pt",
    weapon_model="models/best_weapon.pt",
    pose_model="yolov8n-pose.pt",
    keypoint_threshold=0.3,
    weapon_proximity_px=100
)

results = pipeline.run(source="path/to/video_or_image")
```

---

## Behavioral Analysis Rules

```python
# Hand elevation
if wrist_y < nose_y:  # y origin at top-left
    flag = "hands_raised"

# Knee flexion (crouching)
if angle(hip, knee, ankle) < 100°:
    flag = "crouching"

# Trunk inclination
if abs(dx/dy) > 0.6:  # shoulder vs hip line
    flag = "trunk_inclined"

# Arm extension
if angle(shoulder, elbow, wrist) > 155°:
    flag = "arm_extended"

# Horizontal posture (fallen/lying)
if bbox_width / bbox_height > 1.4:
    flag = "horizontal"
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{mendes2026weapon,
  title     = {Weapon Detection and Behavioral Risk Classification 
               in Surveillance Using Deep Learning},
  author    = {Mendes, Gabriel and Alencar, Jo{\~a}o Pedro and 
               Menezes, Jo{\~a}o Pedro and Fernandes, Rigel P. and
               Figueredo, Cassius and de Souza, Thiago S.},
  booktitle = {XLIV Brazilian Symposium on Telecommunications 
               and Signal Processing (SBrT 2026)},
  year      = {2026},
  address   = {Salvador, BA, Brazil}
}
```

---

## License

This project is for academic purposes. Dataset licensed under [Roboflow Universe Terms](https://roboflow.com/terms).
