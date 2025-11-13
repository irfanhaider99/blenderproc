import os
import shutil
import random

input_path = "/data/BlenderProc1/output_main_v2/ipd/train_pbr"
output_base = "/data/BlenderProc1/output_split"
splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}

scenes = sorted([d for d in os.listdir(input_path) if d.isdigit()])
random.shuffle(scenes)

total = len(scenes)
n_train = int(total * splits['train'])
n_val = int(total * splits['val'])

split_dirs = {
    'train': scenes[:n_train],
    'val': scenes[n_train:n_train + n_val],
    'test': scenes[n_train + n_val:]
}

for split, scene_list in split_dirs.items():
    for scene in scene_list:
        src = os.path.join(input_path, scene)
        dst = os.path.join(output_base, split, scene)
        shutil.copytree(src, dst)

print(f"Split completed: {len(split_dirs['train'])} train, {len(split_dirs['val'])} val, {len(split_dirs['test'])} test")

