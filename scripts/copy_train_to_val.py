# import os
# import random
# import shutil

# # Paths
# TRAIN_DIR = r"D:\fypdataset\muzzle_dataset\train"
# VAL_DIR   = r"D:\fypdataset\muzzle_dataset\val"

# SPLIT_RATIO = 0.3  # 30%

# # Create validation directory if it doesn't exist
# os.makedirs(VAL_DIR, exist_ok=True)

# # List all cow folders
# cow_folders = [
#     d for d in os.listdir(TRAIN_DIR)
#     if os.path.isdir(os.path.join(TRAIN_DIR, d))
# ]

# total_cows = len(cow_folders)
# num_val_cows = int(total_cows * SPLIT_RATIO)

# # Randomly select cow folders for validation
# val_cows = random.sample(cow_folders, num_val_cows)

# print(f"Total cows: {total_cows}")
# print(f"Validation cows (30%): {num_val_cows}\n")

# for cow_id in val_cows:
#     src = os.path.join(TRAIN_DIR, cow_id)
#     dst = os.path.join(VAL_DIR, cow_id)

#     if not os.path.exists(dst):
#         shutil.copytree(src, dst)

#     print(f"Copied folder: {cow_id}")

# print("\n✅ 30% of cow folders copied successfully from train → val")
import os
import random
import shutil

# Paths
TRAIN_DIR = r"D:\fypdataset\muzzle_dataset\train"
VAL_DIR   = r"D:\fypdataset\muzzle_dataset\val1"

SPLIT_RATIO = 0.3  # 30%

# Create val directory if not exists
os.makedirs(VAL_DIR, exist_ok=True)

for cow_id in os.listdir(TRAIN_DIR):
    cow_train_path = os.path.join(TRAIN_DIR, cow_id)

    if not os.path.isdir(cow_train_path):
        continue

    images = [f for f in os.listdir(cow_train_path)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(images) == 0:
        continue

    # Number of images to copy
    num_val = max(1, int(len(images) * SPLIT_RATIO))

    selected_images = random.sample(images, num_val)

    cow_val_path = os.path.join(VAL_DIR, cow_id)
    os.makedirs(cow_val_path, exist_ok=True)

    for img in selected_images:
        src = os.path.join(cow_train_path, img)
        dst = os.path.join(cow_val_path, img)

        if not os.path.exists(dst):  # avoid overwrite
            shutil.copy(src, dst)

    print(f"{cow_id}: copied {len(selected_images)} images to validation")

print("\n✅ 30% images copied successfully from train to val")
