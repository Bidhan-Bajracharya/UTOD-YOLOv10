# train_uodd_yolov10.py
# -------------------------------------------------------------
# Script to verify label normalization and train YOLOv10 model
# Dataset: Underwater-object-detection-dataset (YOLO format)
# -------------------------------------------------------------

from ultralytics import YOLOv10
from pathlib import Path
import os

# === CONFIGURATION ===
DATASET_DIR = "/home/dsai-st125287/project/Underwater-object-detection-dataset"
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
YAML_PATH = "/home/dsai-st125287/project/UTOD-YOLOv10/data/uodd.yaml"  # or wherever your YAML is saved

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

        SEED = 42
        model = YOLOv10("yolov10n.yaml")  # Build YOLOv10n model from YAML

        model.train(
            data=YAML_PATH,          # Path to dataset YAML file
            epochs=100,
            imgsz=640,
            batch=-1,
            device=0,                # GPU 0 on HPC
            pretrained=False,
            workers=0,
            seed=SEED
        )

        print("\n✅ Training complete.")
