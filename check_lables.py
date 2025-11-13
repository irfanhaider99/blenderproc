import os
import json
import cv2

# Modify these paths
scene_dir = "/data/BlenderProc1/output_main_v2/ipd/train_pbr/000000"
rgb_dir = os.path.join(scene_dir, "rgb")
gt_info_path = os.path.join(scene_dir, "scene_gt_info.json")

# Image size
IMG_WIDTH = 1936
IMG_HEIGHT = 1216

# Load bbox annotations
with open(gt_info_path, 'r') as f:
    gt_info = json.load(f)

for frame_id, obj_infos in gt_info.items():
    rgb_path = os.path.join(rgb_dir, f"{int(frame_id):06d}.jpg")
    img = cv2.imread(rgb_path)

    for obj in obj_infos:
        bbox = obj.get("bbox_obj", [])
        if len(bbox) == 4:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("BOP 2D Bounding Boxes", img)
    if cv2.waitKey(0) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

