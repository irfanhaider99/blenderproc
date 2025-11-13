import blenderproc as bproc
import numpy as np
import os

# Initialize BlenderProc
bproc.init()

# Load mesh objects (.ply or .obj)
objs = bproc.loader.load_meshes('./resources/my_objects')

# Apply random materials and disable physics
for obj in objs:
    obj.enable_rigidbody(False)
    obj.random_material()

# Add lighting
light = bproc.types.Light()
light.set_type("AREA")
light.set_location([2, 5, 5])
light.set_energy(300)

# Camera poses — generate 100 random views
for _ in range(100):
    location = np.random.uniform([-1.5, -1.5, 1.5], [1.5, 1.5, 2.5])
    look_at = np.array([0.0, 0.0, 0.0])
    forward_vec = look_at - location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)

# Rendering settings
bproc.renderer.set_output_format(enable_transparency=False)
bproc.renderer.enable_depth_output(False)
bproc.renderer.set_light_bounces(diffuse_bounces=1, glossy_bounces=1, max_bounces=1)
bproc.renderer.set_samples(32)  # You can increase to 64+ if needed for better quality

# Create output folder
output_dir = './output/yolo_dataset_large'
os.makedirs(output_dir, exist_ok=True)

# Render and collect data
data = bproc.renderer.render()

# Save images and annotations in BOP format (YOLO-compatible with conversion)
bproc.writer.write_bop(output_dir, objs, data)

print(" Rendered and annotated 100+ images for YOLO training.")
