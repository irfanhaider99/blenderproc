import os

base_path = "/data/BlenderProc1/Object_detetction/yolov_converstion/labels"
invalid = 0
total = 0

for split in ['train', 'val', 'test']:
    path = os.path.join(base_path, split)
    if not os.path.exists(path):
        continue
    for f in os.listdir(path):
        if not f.endswith('.txt'):
            continue
        file_path = os.path.join(path, f)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                total += 1
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"❌ {f} [Line {idx+1}]: Incorrect number of elements: {parts}")
                    invalid += 1
                    continue
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    if cls < 0 or not all(0 < val <= 1 for val in [x, y, w, h]):
                        print(f"❌ {f} [Line {idx+1}]: Invalid value(s) → {parts}")
                        invalid += 1
                except Exception as e:
                    print(f"❌ {f} [Line {idx+1}]: Error parsing line → {line.strip()}")
                    invalid += 1

print(f"✅ Total labels: {total}, Invalid labels: {invalid}")

