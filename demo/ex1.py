import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Read the CSV
df = pd.read_csv("D:/fypdataset/cattle_labels.csv")

# Folder where images are actually stored
image_folder = "D:/fypdataset/images/"

# Loop through each row and show image
for index, row in df.iterrows():
    # Construct full path
    img_path = os.path.join(image_folder, row['image_name'])

    # Read image
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not load image at: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image with label
    plt.imshow(img)
    plt.title(row['label'])
    plt.axis('off')
    plt.show()
