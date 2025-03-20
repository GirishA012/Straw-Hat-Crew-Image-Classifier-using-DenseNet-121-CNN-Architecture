import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories
DATASET_DIR = "datasets"
AUGMENTED_DIR = "augmented_data"

# Create directory if not exists
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process each crew member's folder
for crew_member in os.listdir(DATASET_DIR):
    member_dir = os.path.join(DATASET_DIR, crew_member.strip())

    # **Ensure it is a valid directory (ignore files like 'desktop.ini')**
    if not os.path.isdir(member_dir) or crew_member.startswith(".") or crew_member.endswith(".ini"):
        continue

    save_dir = os.path.join(AUGMENTED_DIR, crew_member.strip())
    os.makedirs(save_dir, exist_ok=True)

    # Collect only image files (ignoring system files)
    images = [os.path.join(member_dir, img) for img in os.listdir(member_dir) if img.lower().endswith((".jpeg", ".jpg", ".png"))]

    for img_path in tqdm(images, desc=f"Augmenting {crew_member.strip()}"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        num_generated = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=save_dir, save_prefix=crew_member.strip(), save_format="jpeg"):
            num_generated += 1
            if num_generated >= (3000 // len(images)):  # Ensure around 3000 images per member
                break

print("✅ Data augmentation completed successfully!")
