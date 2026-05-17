import json
from pathlib import Path

# -----------------------------
# EDIT THESE PATHS
# -----------------------------
coco_json_path = Path("/home/dsai-st125287/project/RUOD/Environmet_ANN/light/instances_light.json")

subset_dirs = {
    # "Blur": Path("/home/dsai-st125287/project/RUOD/Environment_pic/blur"),
    "Color Distortion": Path("/home/dsai-st125287/project/RUOD/Environment_pic/color"),
    # "Lighting Distortion": Path("/home/dsai-st125287/project/RUOD/Environment_pic/light"),
}

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------------
# LOAD COCO JSON
# -----------------------------
with open(coco_json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# Map filename -> image_id
filename_to_id = {
    Path(img["file_name"]).name: img["id"]
    for img in images
}

# Map image_id -> number of annotations
ann_count_by_image_id = {}
for ann in annotations:
    image_id = ann["image_id"]
    ann_count_by_image_id[image_id] = ann_count_by_image_id.get(image_id, 0) + 1

# -----------------------------
# COUNT SUBSET IMAGES + INSTANCES
# -----------------------------
for subset_name, subset_path in subset_dirs.items():
    image_files = [
        p for p in subset_path.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    image_ids = []
    missing_files = []

    for img_path in image_files:
        filename = img_path.name
        if filename in filename_to_id:
            image_ids.append(filename_to_id[filename])
        else:
            missing_files.append(filename)

    instance_count = sum(
        ann_count_by_image_id.get(image_id, 0)
        for image_id in image_ids
    )

    print(f"\n{subset_name}")
    print(f"Images in folder: {len(image_files)}")
    print(f"Images matched in COCO JSON: {len(image_ids)}")
    print(f"Object instances: {instance_count}")

    if missing_files:
        print(f"WARNING: {len(missing_files)} image files not found in COCO JSON")
        print("First few missing:", missing_files[:10])