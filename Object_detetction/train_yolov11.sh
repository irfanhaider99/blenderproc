#!/bin/bash

# Optional: activate environment if needed
# source ~/anaconda3/bin/activate bproc_env

# Run YOLOv11 training
yolo detect train \
  model=yolov11.pt \
  data=/data/BlenderProc1/output_yolov/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=yolov11-ipd

