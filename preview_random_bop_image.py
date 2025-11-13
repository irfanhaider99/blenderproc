import os
import json
import random
import cv2

def draw_boxes(image, boxes, color=(0, 255, 0), label="obj"):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main(dataset_root):
    scene_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    chosen_scene = random.choice(scene_dirs)
    scene_path = os.path.join(dataset_root, chosen_scene)
    print(f"Previewing: {scene_path}")

    # Load ground truth boxes and info
    gt_file = os.path.join(scene_path, "scene_gt.json")
    gt_info_file = os.path.join(scene_path, "scene_gt_info.json")

    with open(gt_file, 'r') as f:
        gt = json.load(f)
    with open(gt_info_file, 'r') as f:
        gt_info = json.load(f)

    keys = list(gt.keys())
    chosen_key = random.choice(keys)
    img_index = int(chosen_key)

    # Try loading .png or .jpg
    for ext in ['png', 'jpg']:
        rgb_path = os.path.join(scene_path, "rgb", f"{img_index:06d}.{ext}")
        if os.path.exists(rgb_path):
            break
    else:
        raise FileNotFoundError("No RGB image found for index", img_index)

    image = cv2.imread(rgb_path)
    if image is None:
        raise IOError("Failed to load image", rgb_path)

    boxes = []
    for obj in gt[chosen_key]:
        info = gt_info[chosen_key][obj["obj_id"] - 1]  # assumes 1-based IDs
        if "bbox_obj" in info:
            x, y, w, h = info["bbox_obj"]
            boxes.append([x, y, w, h])

    draw_boxes(image, boxes, label="object")

    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("/data/BlenderProc1/output_main_v2/ipd/train_pbr")

