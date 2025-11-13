from ultralytics import YOLO

model = YOLO("/data/BlenderProc1/Object_detetction/local_training/yolov11_finetune_v2/weights/best.pt")

model.train(
    data="/data/BlenderProc1/Object_detetction/ROBOASSIST_v12/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    name="yolov11_finetune_v3",
    project="/data/BlenderProc1/Object_detetction/local_training"
)

