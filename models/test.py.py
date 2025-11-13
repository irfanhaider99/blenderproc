import blenderproc as bproc
import random

# Initialize the BlenderProc scene
bproc.init()

# Load your ply files (change these paths to where your files are located)
object_paths = ["object1.ply", "object2.ply", "object3.ply", "object4.ply"]
objects = [bproc.loader.load_mesh(obj_path) for obj_path in object_paths]

# Set random positions and rotations for each object
for obj in objects:
    obj.set_location([random.uniform(-1, 1), random.uniform(-1, 1), 0])
    obj.set_rotation_euler([random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)])

# Setup camera
camera = bproc.camera.Camera()
camera.set_location([2, 2, 2])
camera.look_at([0, 0, 0])
bproc.camera.add_camera_to_scene(camera)

# Setup lighting
light = bproc.types.Light(type="POINT")
light.set_location([3, 3, 3])
light.set_energy(random.uniform(100, 500))

# Render the scene with images
bproc.render.render()

