import json
import os
import glob

# Change paths if needed
LABELS_DIR = "D:/fypdataset/yolo_dataset/labels/val"
IMAGES_DIR = "D:/fypdataset/yolo_dataset/images/val"

CLASS_MAP = {
    "muzzle": 0
}

def convert(json_path):
    with open(json_path) as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    txt_path = json_path.replace(".json", ".txt")

    with open(txt_path, "w") as out_file:
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            x_min = min(p[0] for p in points)
            y_min = min(p[1] for p in points)
            x_max = max(p[0] for p in points)
            y_max = max(p[1] for p in points)

            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            class_id = CLASS_MAP[label]

            out_file.write(
                f"{class_id} {x_center} {y_center} {width} {height}\n"
            )

# Convert all JSONs
for json_file in glob.glob(os.path.join(LABELS_DIR, "*.json")):
    convert(json_file)

print("✅ Conversion completed!")
