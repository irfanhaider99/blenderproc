import cv2
from ultralytics import YOLO
import os

# === Configuration ===
video_path = "cam1.mp4"  # ✅ Ensure filename is correct
model_path = "/data/BlenderProc1/Object_detetction/local_training/yolov11_finetune_v3/weights/best.pt"
output_path = "output1_detected.avi"
device = "cuda"  # or "cpu"

# === Load YOLOv11 model ===
model = YOLO(model_path)
model.to(device)

# === Load video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

# === Setup video writer ===
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fallback FPS

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height)
)

# === Frame-by-frame detection ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video or cannot read frame.")
        break

    # Inference (BGR works with YOLOv11, RGB not required explicitly)
    results = model(frame, conf=0.25, verbose=False)[0]

    # Draw boxes
    if results.boxes is not None and results.boxes.xyxy is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls = classes[i]
            conf = confs[i]

            # Fallback if model.names is not present
            class_name = model.names[cls] if hasattr(model, "names") else str(cls)
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show and save
    cv2.imshow("YOLOv11 Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Interrupted by user.")
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection complete. Output saved to:", output_path)

