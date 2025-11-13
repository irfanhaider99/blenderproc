from glob import glob
import os

label_files = glob('/data/BlenderProc1/yolov_converstion/labels/*.txt')
class_counts = [0] * 4

for lf in label_files:
    with open(lf, 'r') as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1

print("Object Counts:", class_counts)

