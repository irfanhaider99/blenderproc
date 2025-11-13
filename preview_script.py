import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

def draw_bboxes(img, boxes, color=(0, 255, 0)):
    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img

def preview_sample(folder):
    rgb_dir = os.path.join(folder, "rgb")
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    if not rgb_files:
        raise RuntimeError(f"No .jpg images found in {rgb_dir}")

    rgb_path = os.path.join(rgb_dir, rgb_files[0])
    img_id_str = os.path.splitext(rgb_files[0])[0]
    img_id_int = int(img_id_str.lstrip("0") or "0")

    info_path = os.path.join(folder, "scene_gt_info.json")
    with open(info_path, 'r') as f:
        gt_info = json.load(f)

    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {rgb_path}")

    # Use int or str keys depending on what's in scene_gt_info.json
    if str(img_id_int) in gt_info:
        ann_list = gt_info[str(img_id_int)]
    elif img_id_int in gt_info:
        ann_list = gt_info[img_id_int]
    else:
        raise KeyError(f"Image ID {img_id_int} not found in scene_gt_info.json")

    boxes = [tuple(map(int, ann["bbox_obj"])) for ann in ann_list if "bbox_obj" in ann]
    img = draw_bboxes(img, boxes)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("sample_preview.png")
    plt.show()

# Run it
preview_sample("/data/BlenderProc1/output/ipd/train_pbr/000001")

