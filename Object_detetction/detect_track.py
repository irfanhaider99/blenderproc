from ultralytics import YOLO
import cv2

# Load your fine-tuned YOLOv11 model
model = YOLO("/data/BlenderProc1/Object_detetction/local_training/yolov11_finetune_v2/weights/best.pt")

# Path to BoT-SORT tracker config file
tracker_config = "botsort.yaml" 

# Use webcam or video file as source
#source = 0  # 0 = webcam | "video.mp4" = video file
source= "cam1_objectTracking02.mp4"

# Start tracking
results = model.track(
    source=source,              # webcam or video path
    tracker=tracker_config,     # BoT-SORT config
    show=True,                  # display results live
    save=True,                  # save annotated video
    save_txt=True,              # optional: save results as .txt files
    stream=False                # stream=True if you want manual frame loop
)

