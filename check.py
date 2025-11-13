import open3d as o3d

# Define your PLY file paths
files = [
    "/data/custom_dataset/models/obj_000000.ply",
    "/home/ihaider/BlenderProc/tless/models/obj_000001.ply",
    "/home/ihaider/BlenderProc/tless/models/obj_000002.ply",
    "/home/ihaider/BlenderProc/tless/models/obj_000003.ply"
]

# Try loading the PLY files and checking for errors
for file in files:
    try:
        pcd = o3d.io.read_point_cloud(file)
        print(f"Loaded {file} successfully with {len(pcd.points)} points.")
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"Error loading {file}: {e}")

