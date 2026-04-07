# -------------------------------------------------------------
# YOLOv10 Training Script
# -------------------------------------------------------------

import argparse
from ultralytics import YOLOv10
from pathlib import Path
import os
import random, numpy as np, torch


# -------------------------------------------------------------
# DATASET MAPPING (Shortcodes → Paths)
# -------------------------------------------------------------
DATASETS = {
    "tci": {
        "dir": "/home/dsai-st125287/project/TrashCan/instance_v",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/trashcan_i.yaml"
    },
    "tcm": {
        "dir": "/home/dsai-st125287/project/TrashCan/material_v",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/trashcan_m.yaml"
    },
    "ruod": {
        "dir": "/home/dsai-st125287/project/RUOD_YOLO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod.yaml"
    },
    "ruod_mini": {
        "dir": "/home/dsai-st125287/project/RUOD_YOLO_MINI",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod_mini.yaml"
    },
    "duo": {
        "dir": "/home/dsai-st125287/project/DUO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/duo.yaml"
    },
    "uodd": {
        "dir": "/home/dsai-st125287/project/UODD_YOLO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/uodd.yaml"
    }
}


# -------------------------------------------------------------
# Check label normalization
# -------------------------------------------------------------
def is_valid(v):
    return 0.0 <= v <= 1.0


def check_labels(labels_dir):
    bad = []
    for txt in Path(labels_dir).rglob("*.txt"):
        with open(txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                bad.append((txt, i, ln, "Invalid format"))
                continue

            try:
                cls, x, y, w, h = parts
                x, y, w, h = map(float, (x, y, w, h))
            except:
                bad.append((txt, i, ln, "Non-float values"))
                continue

            if not all(is_valid(v) for v in (x, y, w, h)):
                bad.append((txt, i, ln, "Out of range"))
    return bad


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds", type=str, required=True,
                        help="Dataset shortcode: tci, tcm, ruod, ruod_mini, duo, uodd")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve dataset paths
    # ---------------------------------------------------------
    if args.ds not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{args.ds}'. Choose one of: {list(DATASETS.keys())}"
        )

    DATASET_DIR = DATASETS[args.ds]["dir"]
    YAML_PATH = DATASETS[args.ds]["yaml"]
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")

    print(f"Using dataset '{args.ds}'")
    print(f"DATASET_DIR = {DATASET_DIR}")
    print(f"YAML        = {YAML_PATH}")

    # ---------------------------------------------------------
    # Apply seed
    # ---------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Check labels
    # ---------------------------------------------------------
    print("=== Checking label normalization ===")
    bad_entries = check_labels(LABELS_DIR)

    if bad_entries:
        print("⚠️ Found invalid normalized values:")
        for path, line_num, content, reason in bad_entries:
            print(f"[{reason}] {path} (line {line_num}): {content}")
        print(f"\nTotal issues: {len(bad_entries)}")
        return

    print("✅ All labels OK")
    print("\n=== Starting YOLOv10 Training ===")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    model = YOLOv10(args.model)

    model.train(
        data=YAML_PATH,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        pretrained=False,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        degrees=15,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.3,
        # mosaic=0.0, # Using 0.0 to disable mosaic augmentation
        # mixup=0.0, # Using 0.0 to disable mixup augmentation
        workers=0,
        seed=args.seed,
        name=args.name
    )

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()

# python train.py \
#     --ds ruod \
#     --model yolov10nDCN.yaml \
#     --name ruod_dcnv4_v3 \
