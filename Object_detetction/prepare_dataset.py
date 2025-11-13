# prepare_dataset.py
import os
from glob import glob
import random

image_dir = "/data/BlenderProc1/Object_detetction/yolov_converstion/images"
output_dir = "/data/BlenderProc1/Object_detetction/yolov_converstion"

# Search recursively in all subdirectories
images = sorted(glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
random.shuffle(images)

n = len(images)
train, val, test = images[:int(n*0.8)], images[int(n*0.8):int(n*0.9)], images[int(n*0.9):]

for name, split in zip(["train", "val", "test"], [train, val, test]):
    with open(os.path.join(output_dir, f"{name}.txt"), "w") as f:
        for img in split:
            f.write(os.path.relpath(img, output_dir) + "\n")

print(f"Dataset split complete: {len(train)} train, {len(val)} val, {len(test)} test")

