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
        object_model_unit='mm',
        sample_objects=True,
        num_of_objs_to_sample=10
    )
    for idx, obj in enumerate(objs):
        obj.set_cp("category_id", idx)

    for obj in objs:
        bbox = obj.get_bound_box()
        size = np.max(bbox.max() - bbox.min())
        if size < 0.05:
            scale = 0.05 / size
            obj.set_scale([scale, scale, scale])

        obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.set_shading_mode('auto')
        try:
            mat = obj.get_materials()[0]
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

    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4, check_object_interval=1)

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

        # ========== SAVE YOLO LABELS ==========
os.makedirs(os.path.join(args.output_dir, "labels/yolo"), exist_ok=True)
label_path = os.path.join(args.output_dir, "labels/yolo", f"{pose_counter:06d}.txt")

with open(label_path, 'w') as f:
    for obj in placed_objs:
        class_id = obj.get_cp("category_id") if obj.has_cp("category_id") else 0
        bbox = obj.get_visible_bounding_box_2d()
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height
            if box_width > 0 and box_height > 0:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


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
