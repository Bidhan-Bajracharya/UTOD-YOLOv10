import argparse
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor

# ---------------------------------------------------------
# Path mappings
# ---------------------------------------------------------
MODEL_PATHS = {
    "base": "/home/dsai-st125287/project/UTOD-YOLOv10/runs/detect/ruod_bs_v2/weights/best.pt",
    "custom": "/home/dsai-st125287/project/UTOD-YOLOv10/runs/detect/ruod_dbw_v12/weights/best.pt",
}

DATASETS = {
    "blur": "/home/dsai-st125287/project/RUOD_BLUR/images",
    "light": "/home/dsai-st125287/project/RUOD_LIGHT/images",
    "color": "/home/dsai-st125287/project/RUOD_COLOR/images",
}


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 Inference for Qualitative Results")

    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "custom"],
        required=True,
        help="Model type to evaluate",
    )

    parser.add_argument(
        "--data",
        type=str,
        choices=["blur", "light", "color"],
        required=True,
        help="RUOD evaluation dataset variant",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Output folder name",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------
    model_path = MODEL_PATHS[args.model]
    dataset_path = DATASETS[args.data]

    print(f"Using model     : {args.model}")
    print(f"Model path      : {model_path}")
    print(f"Dataset variant : {args.data}")
    print(f"Image folder    : {dataset_path}")
    print(f"Run name        : {args.name}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = YOLO(model_path)

    # ---------------------------------------------------------
    # Configure predictor
    # ---------------------------------------------------------
    cfg = get_cfg()

    cfg.model = model_path
    cfg.source = dataset_path
    cfg.imgsz = args.imgsz
    cfg.device = args.device
    cfg.conf = args.conf

    cfg.project = "/home/dsai-st125287/project/runs-inf/inference"
    cfg.name = args.name

    # ---------------------------------------------------------
    # Run inference using YOLOv10 predictor
    # ---------------------------------------------------------
    predictor = YOLOv10DetectionPredictor(cfg=cfg)
    predictor()

    print("Inference completed.")


if __name__ == "__main__":
    main()