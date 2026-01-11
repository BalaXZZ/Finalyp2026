import os
import random
import shutil

# ==============================
# PATHS (your exact paths)
# ==============================
SOURCE_DIR = "D:/fypdataset/dataset/images"
TRAIN_DIR = "D:/fypdataset/yolo_dataset/images/train"
VAL_DIR = "D:/fypdataset/yolo_dataset/images/val"

# ==============================
# CREATE DIRECTORIES IF NOT EXIST
# ==============================
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==============================
# LOAD IMAGE FILES
# ==============================
images = [
    img for img in os.listdir(SOURCE_DIR)
    if img.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"📂 Total images found: {len(images)}")

# ==============================
# SHUFFLE & SPLIT
# ==============================
random.shuffle(images)

split_index = int(0.8 * len(images))  # 80% train
train_images = images[:split_index]
val_images = images[split_index:]

# ==============================
# COPY FILES
# ==============================
for img in train_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TRAIN_DIR, img)
    )

for img in val_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(VAL_DIR, img)
    )

print(f"✅ Training images: {len(train_images)}")
print(f"✅ Validation images: {len(val_images)}")
print("🎉 Dataset split completed successfully!")
