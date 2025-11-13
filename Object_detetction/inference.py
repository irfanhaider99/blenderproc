from ultralytics import YOLO
import cv2
import os

# Load the trained YOLO model
model = YOLO("runs/detect/yolov11n-custom14/weights/best.pt")

# Load a single test image
img_file = "/data/BlenderProc1/yolov_converstion/images/test/000000.jpg"  
img = cv2.imread(img_file)

# Run inference
results = model(img_file)  # You can also pass 'img' directly

# Get class names from model
names = model.names  # e.g., {0: 'obj_000001', 1: 'obj_000002', ...}

# Draw detections
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        name = names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{name} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Show and save the result
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("detection_result.jpg", img)  # Optional: save the output

