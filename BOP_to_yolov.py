import os
import json
import shutil
from tqdm import tqdm
from PIL import Image

# Paths
bop_base_path = "/data/BlenderProc1/output_split"  # Where your BOP-split data is
yolo_output_path = "/data/BlenderProc1/yolov_converstion"  # Final YOLO data location
splits = ["train", "val", "test"]

# Create YOLO folder structure
for split in splits:
    os.makedirs(os.path.join(yolo_output_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(yolo_output_path, "labels", split), exist_ok=True)

def convert_bop_to_yolo(bbox_visib, img_w, img_h):
    """Convert BOP bbox to YOLO format (normalized)."""
    x, y, w, h = bbox_visib
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

# Track stats
image_count = 0
split_counts = {split: 0 for split in splits}
class_ids = set()

# Process each split
for split in splits:
    split_dir = os.path.join(bop_base_path, split)
    if not os.path.isdir(split_dir):
        print(f"⚠️ Skipping missing split directory: {split_dir}")
        continue

    scenes = sorted(os.listdir(split_dir))
    for scene_id in scenes:
        scene_path = os.path.join(split_dir, scene_id)
        rgb_path = os.path.join(scene_path, "rgb")
        gt_path = os.path.join(scene_path, "scene_gt.json")
        info_path = os.path.join(scene_path, "scene_gt_info.json")

        if not os.path.isdir(rgb_path) or not os.path.exists(gt_path) or not os.path.exists(info_path):
            print(f"⚠️ Skipping incomplete scene: {split}/{scene_id}")
            continue

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(info_path, 'r') as f:
            info_data = json.load(f)

        for img_id_str in gt_data:
            img_id = int(img_id_str)
            img_file = f"{img_id:06d}.jpg"
            img_path = os.path.join(rgb_path, img_file)
            if not os.path.exists(img_path):
                continue

            # Get image size
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # Copy image to YOLO folder
            out_img_path = os.path.join(yolo_output_path, "images", split, img_file)
            shutil.copy(img_path, out_img_path)

            # Write YOLO label file
            label_file = os.path.join(yolo_output_path, "labels", split, f"{img_id:06d}.txt")
            with open(label_file, 'w') as lf:
                for obj_ann, obj_info in zip(gt_data[img_id_str], info_data[img_id_str]):
                    cls_id = obj_ann["obj_id"] - 1  # 0-based class ID
                    class_ids.add(cls_id)
                    bbox = obj_info["bbox_visib"]
                    x_center, y_center, w, h = convert_bop_to_yolo(bbox, img_w, img_h)
                    lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            image_count += 1
            split_counts[split] += 1

# Write YOLO data.yaml
class_ids = sorted(list(class_ids))
names = [f"obj_{i+1:06d}" for i in class_ids]
data_yaml_path = os.path.join(yolo_output_path, "data.yaml")
with open(data_yaml_path, 'w') as f:
    f.write(f"train: {yolo_output_path}/images/train\n")
    f.write(f"val: {yolo_output_path}/images/val\n")
    f.write(f"test: {yolo_output_path}/images/test\n")
    f.write(f"nc: {len(class_ids)}\n")
    f.write(f"names: {names}\n")

# Print summary
print("\n✅ YOLO Conversion Completed")
print(f"Total images converted: {image_count}")
for split in splits:
    print(f"{split.capitalize()}: {split_counts[split]} images")
print(f"\nYOLO dataset written to: {yolo_output_path}")
print(f"Labels: labels/<split>/*.txt, Images: images/<split>/*.jpg")
print(f"data.yaml path: {data_yaml_path}")

