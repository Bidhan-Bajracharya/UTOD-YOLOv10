import argparse
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.cfg import get_cfg
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator

# ---------------------------------------------------------
# Path mappings
# ---------------------------------------------------------
MODEL_PATHS = {
    "base": "/home/dsai-st125287/project/UTOD-YOLOv10/runs/detect/ruod_bs_v2/weights/best.pt",
    "custom": "/home/dsai-st125287/project/UTOD-YOLOv10/runs/detect/ruod_dbw_v12/weights/best.pt",
}

DATA_YAMLS = {
    "blur": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod_blur.yaml",
    "light": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod_light.yaml",
    "color": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod_color.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 RUOD validation")

    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "custom"],
        required=True,
        help="Model type to evaluate: base or custom",
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
        help="Name of the validation run",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------
    model_path = MODEL_PATHS[args.model]
    data_yaml = DATA_YAMLS[args.data]

    print(f"Using model     : {args.model}")
    print(f"Model path      : {model_path}")
    print(f"Dataset variant : {args.data}")
    print(f"YAML path       : {data_yaml}")
    print(f"Run name        : {args.name}")

    # ---------------------------------------------------------
    # Load model & validate
    # ---------------------------------------------------------
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        validator=YOLOv10DetectionValidator,
    )


if __name__ == "__main__":
    main()

# python validate_ruod.py \
#     --model base \
#     --data blur \
#     --name ruod_blur_eval_bs

# print(metrics.box.map)
# print(metrics.box.map50)
# print(metrics.box.precision)
# print(metrics.box.recall)
