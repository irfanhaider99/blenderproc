import blenderproc as bproc
import argparse
import os
import numpy as np
from mathutils import Vector
import blenderproc as bproc
import argparse
import os
import numpy as np

# ========== ARG PARSING ==========
parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path')
parser.add_argument('bop_dataset_name')
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures")
parser.add_argument('output_dir')
parser.add_argument('--N', type=int, default=4, help="Number of object-sampling iterations")
args = parser.parse_args()

bproc.init()

# Camera settings
K = np.array([[2346.1826, 0.0,       956.1507],
              [0.0,       2351.0242, 613.65509],
              [0.0,       0.0,       1.0]])
width = 1936
height = 1216
bproc.camera.set_resolution(width, height)
bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

# Build room
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2,2,1]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[0,-2,2], rotation=[-1.570796,0,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[0, 2,2], rotation=[ 1.570796,0,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[2, 0,2], rotation=[0,-1.570796,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[-2,0,2], rotation=[0, 1.570796,0])
]

# Emissive light plane
light_plane = bproc.object.create_primitive('PLANE', scale=[3,3,1], location=[0,0,10])
light_plane.set_name("light_plane")

# Point light
light_point = bproc.types.Light()

# Renderer settings
bproc.renderer.enable_depth_output(activate_antialiasing=True)
bproc.renderer.set_max_amount_of_samples(128)

# Load and randomize objects
all_colors = []
all_depths = []
pose_counter = 0
ANGLES = 4
ROTATIONS = 2

for iteration in range(args.N):
    bproc.camera.camera_poses = []
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    chosen_tex = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(chosen_tex)

    lm = bproc.material.create(f"light_mat_{iteration}")
    lm.make_emissive(
        emission_strength=np.random.uniform(3,5),
        emission_color=np.random.uniform([0.7,0.7,0.7,1.0],[0.9,0.9,0.9,1.0])
    )
    light_plane.replace_materials(lm)
    light_plane.set_rotation_euler([0,0,0])
    light_point.set_energy(np.random.uniform(120,180))
    light_point.set_color(np.random.uniform([0.7,0.7,0.7],[1.0,1.0,1.0]))
    light_point.set_location(bproc.sampler.shell(center=[0,0,0], radius_min=1.2, radius_max=1.4, elevation_min=40, elevation_max=70))

    objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(args.bop_parent_path, args.bop_dataset_name),
        mm2m=True,
        sample_objects=True,
        num_of_objs_to_sample=10
    )

    for obj in objs:
        bbox = obj.get_bound_box()
        size = np.max(bbox.max() - bbox.min())
        if size < 0.05:
            scale = 0.05 / size
            obj.set_scale([scale, scale, scale])

        obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.set_shading_mode('auto')
        mat = obj.get_materials()[0]
        try:
            mat.set_principled_shader_value("Base Color", np.random.uniform(0.4, 0.7, 3).tolist() + [1.0])
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.4, 0.8))
            mat.set_principled_shader_value("Specular", np.random.uniform(0.1, 0.3))
        except Exception as e:
            print(f" Could not set shader values for {obj.get_name()}: {e}")

    placed_objs = []
    for obj in objs:
        tries = 0
        while tries < 50:
            loc = np.random.uniform([-0.1, -0.1, 0.05], [0.01, 0.01, 0.15])
            if all(np.linalg.norm(loc - o.get_location()) > 0.07 for o in placed_objs):
                obj.set_location(loc)
                obj.set_rotation_euler(bproc.sampler.uniformSO3())
                placed_objs.append(obj)
                break
            tries += 1
        if len(placed_objs) < 4:
            print(" Less than 4 objects placed, skipping this iteration.")
            continue

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, 
                                                      max_simulation_time=4,
                                                      check_object_interval= 1)

    for obj in placed_objs:
        loc = obj.get_location()
        loc[2] = max(loc[2], 0.05)
        obj.set_location(loc)

    for _ in range(ANGLES):
        bvh = bproc.object.create_bvh_tree_multi_objects(placed_objs)
        poi = bproc.object.compute_poi(placed_objs)
        cam_loc = bproc.sampler.shell(center=poi, radius_min=0.7, radius_max=1.0, elevation_min=25, elevation_max=50)
        cam_rot = bproc.camera.rotation_from_forward_vec(poi - cam_loc)
        cam_pose = bproc.math.build_transformation_mat(cam_loc, cam_rot)

        if not bproc.camera.perform_obstacle_in_view_check(cam_pose, {"min": 0.25}, bvh):
            continue

        bproc.camera.add_camera_pose(cam_pose)

        for _ in range(ROTATIONS):
            axis = np.random.choice([0,1,2])
            for obj in placed_objs:
                eul = obj.get_rotation_euler()
                eul[axis] += np.pi / 45
                obj.set_rotation_euler(eul)

        data = bproc.renderer.render()
        all_colors.append(data["colors"][-1])
        all_depths.append(data["depth"][-1])
        print(f"✅ Frame {pose_counter} rendered.")
        pose_counter += 1

if len(all_colors) == 0:
    print("❌ No frames rendered — skipping BOP write.")
    exit()

# ========== WRITE BOP ONCE ==========
bproc.writer.write_bop(
    args.output_dir,
    dataset=args.bop_dataset_name,
    depths=all_depths,
    colors=all_colors,
    color_file_format="JPEG",
    ignore_dist_thres=10,
    frames_per_chunk=ROTATIONS
)
# ========== ARG PARSING ==========
parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path')
parser.add_argument('bop_dataset_name')
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures")
parser.add_argument('output_dir')
parser.add_argument('--N', type=int, default=4, help="Number of object-sampling iterations")
args = parser.parse_args()

# ========== INIT BLENDERPROC ==========
bproc.init()

import bpy
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = [0.05, 0.05, 0.05, 1.0]  # RGB + Alpha
bg.inputs[1].default_value = 1.0  # Strength

# ========== CAMERA SETTINGS ==========
K = np.array([
    [2346.1826, 0.0, 956.1507],
    [0.0, 2351.0242, 613.65509],
    [0.0, 0.0, 1.0]
])
width, height = 1936, 1216
bproc.camera.set_resolution(width, height)
bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

# ========== SCENE SETUP ==========
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=1.0, linear_damping=0.9, angular_damping=0.9)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name("light_plane")

# ========== RENDERER SETTINGS ==========
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.set_output_format(enable_transparency=False)

# ========== UTILITIES ==========
def sample_pose_with_distance(obj, placed_objs, min_dist=0.02, max_tries=100):
    for _ in range(max_tries):
        loc = np.random.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.25])
        if all(np.linalg.norm(loc - p.get_location()) >= min_dist for p in placed_objs):
            obj.set_location(loc)
            obj.set_rotation_euler(bproc.sampler.uniformSO3())
            return True
    print(f"❌ Failed to place {obj.get_name()} with spacing {min_dist}")
    return False

# ========== MAIN LOOP ==========
all_colors, all_depths = [], []
pose_counter = 0
ROTATIONS, ANGLES = 2, 4

for iteration in range(args.N):
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    if not cc_textures:
        raise RuntimeError(f"No textures found in {args.cc_textures_path}")

    chosen_tex = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(chosen_tex)

    lm = bproc.material.create("bright_emission")
    lm.make_emissive(emission_strength=np.random.uniform(5, 12), emission_color=np.random.uniform([0.8]*4, [1.0]*4))
    light_plane.replace_materials(lm)

    # Lights
    for loc in [
        np.random.uniform([-1, -1, 1], [1, 1, 3]),
        np.random.uniform([-1, 1, 1], [1, 2, 3]),
        np.random.uniform([-2, -2, 0.1], [2, 2, 0.5])
    ]:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(loc.tolist())
        light.set_energy(np.random.uniform(50, 300))
        light.set_color(np.random.uniform([0.8, 0.8, 0.8], [1.0, 1.0, 1.0]).tolist())

    # Load BOP objects
    bproc.utility.reset_keyframes()
    objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(args.bop_parent_path, args.bop_dataset_name),
        object_model_unit='mm', sample_objects=True, num_of_objs_to_sample=15
    )

    placed_objs = []
    for o in objs:
        o.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        o.set_shading_mode('auto')
        mat = o.get_materials()[0]
        base_color = np.random.uniform(0.2, 0.9, 3)
        mat.set_principled_shader_value("Base Color", base_color.tolist() + [1])

        try:
            if hasattr(mat, "get_principled_shader_input_names"):
                if "Roughness" in mat.get_principled_shader_input_names():
                    mat.set_principled_shader_value("Roughness", np.random.uniform(0.1, 0.9))
                if "Specular" in mat.get_principled_shader_input_names():
                    mat.set_principled_shader_value("Specular", np.random.uniform(0.1, 0.6))
                if "Metallic" in mat.get_principled_shader_input_names():
                    mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 0.5))
        except Exception as e:
            print(f"⚠️ Could not modify shader for '{mat.get_name()}': {e}")

        if sample_pose_with_distance(o, placed_objs):
            placed_objs.append(o)

    # Add distractor objects (random primitives)
    distractors = []
    for _ in range(np.random.randint(3, 6)):
        prim_type = np.random.choice(['CUBE', 'SPHERE', 'CONE'])
        prim = bproc.object.create_primitive(prim_type)
        prim.set_location(np.random.uniform([-0.2, -0.2, 0.02], [0.2, 0.2, 0.2]))
        prim.set_rotation_euler(bproc.sampler.uniformSO3())
        prim.set_scale(np.random.uniform(0.01, 0.05, 3).tolist())
        mat = bproc.material.create(f"distractor_mat_{_}")
        mat.set_principled_shader_value("Base Color", np.random.uniform(0.3, 0.9, 3).tolist() + [1])
        prim.replace_materials(mat)
        distractors.append(prim)

    bproc.object.simulate_physics_and_fix_final_poses(3, 5, check_object_interval=1, substeps_per_frame=10, solver_iters=10)

    filtered_objs = [o for o in placed_objs if o.get_location()[2] > 0.01]
    if not filtered_objs:
        print("⚠️ No valid objects found — skipping.")
        continue

    for o in filtered_objs:
        loc = o.get_location()
        if loc[2] < 0.02:
            loc[2] = 0.02
            o.set_location(loc)

    print(f"✅ Proceeding with {len(filtered_objs)} objects.")

    cam_attempts, cam_angles = 0, 0
    while cam_angles < ANGLES and cam_attempts < 50:
        bvh = bproc.object.create_bvh_tree_multi_objects(filtered_objs)
        poi = bproc.object.compute_poi(filtered_objs)

# Sample camera pose that looks at all objects
cam_poses = bproc.camera.sample_poses_looking_at_poi(
    poi,
    number_of_samples=ANGLES,
    distance_range=[0.6, 0.8],
    elevation_range=[15, 45],
    azimuth_range=[-180, 180]
)

for cam_pose in cam_poses:
    if not bproc.camera.perform_obstacle_in_view_check(cam_pose, {"min": 0.2}, bvh):
        continue

    for _ in range(ROTATIONS):
        axis = np.random.choice([0, 1, 2])
        for o in filtered_objs:
            eul = o.get_rotation_euler()
            eul[axis] += np.pi / 45
            o.set_rotation_euler(eul)

    bproc.camera.add_camera_pose(cam_pose)
    data = bproc.renderer.render()
    all_colors.append(data["colors"][-1])
    all_depths.append(data["depth"][-1])
    print(f"✅ Frame {pose_counter} rendered.")
    pose_counter += 1

if not all_colors:
    print("❌ No frames rendered — exiting.")
    exit()

# ========== WRITE BOP ==========
bproc.writer.write_bop(
    args.output_dir,
    dataset=args.bop_dataset_name,
    depths=all_depths,
    colors=all_colors,
    color_file_format="JPEG",
    ignore_dist_thres=10,
    frames_per_chunk=ROTATIONS
)  # Add segmentations if needed
