import os
import json

data_yaml = {
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 4,
    "names": ["object_0", "object_1", "object_2", "object_3"]
}

with open("/data/BlenderProc1/Object_detetction/output_yolov/data.yaml", 'w') as f:
    yaml_str = json.dumps(data_yaml, indent=4).replace('"', '')
    f.write(yaml_str)

print("✅ data.yaml regenerated successfully.")

