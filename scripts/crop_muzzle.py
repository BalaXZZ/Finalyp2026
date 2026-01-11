# from ultralytics import YOLO
# import cv2
# import os
# import pandas as pd
# df = pd.read_csv("D:\fypdataset\dataset\cattle_labels.csv")
# label_map = dict(zip(df.image_name, df.label))
# IMAGE_ROOT = "yolo_dataset/images"
# OUTPUT_ROOT = "muzzle_dataset"
# model = YOLO("")

# splits = ["train", "val"]
# for split in splits:
#     image_dir = os.path.join(IMAGE_ROOT, split)

#     for img_name in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, img_name)

#         if img_name not in label_map:
#             continue

#         cow_label = label_map[img_name]
#         save_dir = os.path.join(OUTPUT_ROOT, split, cow_label)
#         os.makedirs(save_dir, exist_ok=True)

#         # Load image
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         # YOLO inference
#         results = model(img, conf=0.3)

#         if len(results[0].boxes) == 0:
#             continue

#         # Take best box
#         box = results[0].boxes[0].xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = map(int, box)

#         muzzle = img[y1:y2, x1:x2]

#         if muzzle.size == 0:
#             continue

#         save_path = os.path.join(save_dir, img_name)
#         cv2.imwrite(save_path, muzzle)
# ---------------------------------------------
import os
import cv2
from ultralytics import YOLO

# ---------------- PATHS ----------------
YOLO_MODEL_PATH = r"D:\fypdataset\runs\detect\train2\weights\best.pt"

IMAGE_ROOT = r"D:\fypdataset\yolo_dataset\images"
OUTPUT_ROOT = r"D:\fypdataset\muzzle_dataset"

# ---------------------------------------
model = YOLO(YOLO_MODEL_PATH)

splits = ["train", "val"]

for split in splits:
    image_dir = os.path.join(IMAGE_ROOT, split)
    output_split_dir = os.path.join(OUTPUT_ROOT, split)

    os.makedirs(output_split_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # 🔍 Run YOLO inference
        results = model(image, conf=0.4)

        # Extract cow ID from filename (example: cow12.png → Cow_12)
        cow_id = img_name.lower().replace(".png", "").replace(".jpg", "")
        cow_id = cow_id.replace("cow", "Cow_")

        cow_folder = os.path.join(output_split_dir, cow_id)
        os.makedirs(cow_folder, exist_ok=True)

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            muzzle_crop = image[y1:y2, x1:x2]

            if muzzle_crop.size == 0:
                continue

            save_path = os.path.join(
                cow_folder,
                f"{cow_id}_muzzle_{i}.png"
            )

            cv2.imwrite(save_path, muzzle_crop)

print("✅ Muzzle cropping completed successfully!")
