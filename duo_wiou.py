# duo_wiou.py
# -------------------------------------------------------------
# Script to verify label normalization and train YOLOv10 model
# Dataset: DUO (YOLO format)
# -------------------------------------------------------------

from ultralytics import YOLOv10
from pathlib import Path
import os
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# === CONFIGURATION ===
DATASET_DIR = "/home/dsai-st125287/project/DUO"
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
YAML_PATH = "/home/dsai-st125287/project/UTOD-YOLOv10/data/duo.yaml" 

# === CHECK LABEL NORMALIZATION ===
def is_valid(v):
    return 0.0 <= v <= 1.0

def check_labels(labels_dir):
    bad_entries = []
    for txt in Path(labels_dir).rglob("*.txt"):
        with open(txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                bad_entries.append((txt, i, ln, "Invalid format"))
                continue
            try:
                cls, x, y, w, h = parts
                x, y, w, h = map(float, (x, y, w, h))
            except:
                bad_entries.append((txt, i, ln, "Non-float values"))
                continue
            if not all(is_valid(v) for v in (x, y, w, h)):
                bad_entries.append((txt, i, ln, "Out of range"))
    return bad_entries


if __name__ == "__main__":
    print("=== Checking label normalization ===")
    bad_entries = check_labels(LABELS_DIR)
    if bad_entries:
        print("⚠️ Found invalid normalized values:")
        for path, line_num, content, reason in bad_entries:
            print(f"[{reason}] {path} (line {line_num}): {content}")
        print(f"\nTotal issues found: {len(bad_entries)}")
        print("Fix these before training!")
    else:
        print("✅ All label values are clamped to [0,1].")
        print("\n=== Starting YOLOv10 Training ===")

        model = YOLOv10("yolov10n.yaml")  # Build YOLOv10n model from YAML

        model.train(
            data=YAML_PATH,          # Dataset YAML file
            epochs=200,              # Extended epochs for better convergence
            imgsz=640,               # Input image size
            batch=32,                # Fixed batch size for reproducibility
            device=0,                # Use GPU 0 on HPC
            pretrained=False,        # Train from scratch (for fair comparison)
            optimizer="SGD",         # Default YOLO optimizer
            lr0=0.01,                # Initial learning rate
            momentum=0.937,          # Standard YOLO momentum
            weight_decay=0.0005,     # Regularization term
            degrees=15,              # Rotation ±15°
            fliplr=0.5,              # Horizontal flip probability
            flipud=0.2,              # Vertical flip probability
            hsv_h=0.02,              # Hue jitter
            hsv_s=0.4,               # Saturation jitter
            hsv_v=0.3,               # Brightness jitter
            workers=0,               # Safe for HPC single-process setup
            seed=SEED,                # Global seed for reproducibility
            name="duo_wiou"            # Experiment name
        )

        print("\n✅ Training complete.")
