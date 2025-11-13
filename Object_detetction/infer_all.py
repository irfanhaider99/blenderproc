import glob
import os
import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/yolov11n-custom21/weights/best.pt")
names = model.names  # Class names dictionary

# Path to your test images
test_dir = "/data/BlenderProc1/yolov_converstion/images/test"
image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))

# Create output folder (optional)
output_dir = "predicted_images"
os.makedirs(output_dir, exist_ok=True)

# Run inference on each image
for img_path in image_paths:
    img = cv2.imread(img_path)
    results = model(img_path)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save result to output folder
    out_file = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
    cv2.imwrite(out_file, img)

print("✅ Inference complete. Output saved in:", output_dir)

