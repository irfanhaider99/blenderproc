import os

label_dir = "/data/BlenderProc1/Object_detetction/yolov_converstion/labels"
total_invalid = 0
total_empty = 0

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(label_dir, split)
    if not os.path.exists(split_dir):
        continue

    for file in os.listdir(split_dir):
        if not file.endswith(".txt"):
            continue
        file_path = os.path.join(split_dir, file)
        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    total_invalid += 1
                    continue
                try:
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if cls < 0 or any(c <= 0.0 or c > 1.0 for c in coords):
                        total_invalid += 1
                        continue
                    new_lines.append(f"{cls} {' '.join(f'{c:.6f}' for c in coords)}")
                except:
                    total_invalid += 1
                    continue
        if new_lines:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        else:
            os.remove(file_path)
            total_empty += 1
            print(f"❌ Removed empty label: {file_path}")

print(f"✅ Cleaning complete. Removed {total_invalid} bad lines, {total_empty} empty label files.")

