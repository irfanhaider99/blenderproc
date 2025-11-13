import blenderproc as bproc
import argparse
import os
import numpy as np

# ========== ARGUMENT PARSING ==========
parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path')
parser.add_argument('bop_dataset_name')
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures")
parser.add_argument('output_dir')
parser.add_argument('--N', type=int, default=10)
args = parser.parse_args()

bproc.init()

# ========== ROOM CREATION ==========
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-np.pi/2, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[np.pi/2, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -np.pi/2, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, np.pi/2, 0])
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX')
    plane.set_cp("category_id", None)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name("light_plane")
light_plane.set_cp("category_id", None)
light_point = bproc.types.Light()

# ========== RENDER SETTINGS ==========
bproc.renderer.set_max_amount_of_samples(64)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by="instance")

# ========== POSE SAMPLING ==========
def sample_pose(obj):
    obj.set_scale([np.random.uniform(0.05, 0.09)] * 3)
    obj.set_location(np.random.uniform([-0.25, -0.25, 0.05], [0.25, 0.25, 0.3]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

pose_counter = 0
ROTATIONS = 3
ANGLES = 5

# ========== MAIN LOOP ==========
for iteration in range(args.N):
    bproc.utility.reset_keyframes()

    # Apply random wall textures
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    for plane in room_planes:
        plane.replace_materials(np.random.choice(cc_textures))

    # Emissive light
    light_mat = bproc.material.create("light_mat")
    light_mat.make_emissive(np.random.uniform(10, 20), [1.0, 1.0, 1.0, 1.0])
    light_plane.replace_materials(light_mat)

    # Point light
    light_point.set_energy(np.random.uniform(200, 400))
    light_point.set_color([1.0, 1.0, 1.0])
    light_point.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=0.5, radius_max=0.8, elevation_min=20, elevation_max=80))

    # Load and sample objects
    objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(args.bop_parent_path, args.bop_dataset_name),
        object_model_unit='mm',
        #mm2m=True,
        sample_objects=True,
        num_of_objs_to_sample=np.random.randint(3, 6)
    )

    for i, obj in enumerate(objs):
        obj.set_cp("category_id", i + 1)
        obj.enable_rigidbody(True, collision_shape='MESH')
        obj.set_shading_mode('auto')
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1))

    bproc.object.sample_poses(objs, 
                              sample_pose_func=sample_pose, 
                              max_tries=1000,
                              #skip_inside_check=True
                              )


    # Remove fallen objects
    objs = [
        obj for obj in objs
        if obj.get_bound_box() is not None and min([v[2] for v in obj.get_bound_box().tolist()]) > -0.01
    ]
    if not objs:
        print(f"[{iteration}] Skipping scene, all objects fell or invalid.")
        continue

   # ========== CAMERA LOOP ==========
successful_views = 0
attempts = 0

while successful_views < ANGLES and attempts < 30:
    attempts += 1
    bvh = bproc.object.create_bvh_tree_multi_objects(objs)
    poi = bproc.object.compute_poi(objs)
    cam_loc = bproc.sampler.shell(
        center=poi,
        radius_min=0.4,
        radius_max=0.6,
        elevation_min=15,
        elevation_max=65,
        azimuth_min=-180,
        azimuth_max=180
    )
    cam_rot = bproc.camera.rotation_from_forward_vec(poi - cam_loc)
    cam_pose = bproc.math.build_transformation_mat(cam_loc, cam_rot)

    if not bproc.camera.perform_obstacle_in_view_check(cam_pose, {"min": 0.2}, bvh):
        continue

    bproc.camera.add_camera_pose(cam_pose)
    data = bproc.renderer.render()

    if "colors" in data and "depth" in data and len(data["colors"]) > 0 and len(data["depth"]) > 0:
        mean_brightness = np.mean(data["colors"][-1])

        if mean_brightness < 0.05:
            print(f"⚠️ Too dark (mean={mean_brightness:.4f}) — skipping frame.")
            bproc.camera.remove_last_added_camera_pose()
            continue

        bproc.writer.write_bop(
            args.output_dir,
            dataset=args.bop_dataset_name,
            depths=[data["depth"][-1]],
            colors=[data["colors"][-1]],
            color_file_format="JPEG",
            append_to_existing_output=True
        )
        pose_counter += 1
        successful_views += 1

        # ========== ROTATED VIEWS ==========
    for _ in range(ROTATIONS):
       for obj in objs:
        eul = obj.get_rotation_euler()
        eul[np.random.randint(3)] += np.pi / 36
        obj.set_rotation_euler(eul)

    bproc.camera.add_camera_pose(cam_pose)
    rotated_data = bproc.renderer.render()

    if (
        "colors" in rotated_data and "depth" in rotated_data
        and len(rotated_data["colors"]) > 0 and len(rotated_data["depth"]) > 0
        and np.mean(rotated_data["colors"][-1]) > 0.05
    ):
        bproc.writer.write_bop(
            args.output_dir,
            dataset=args.bop_dataset_name,
            depths=[rotated_data["depth"][-1]],
            colors=[rotated_data["colors"][-1]],
            color_file_format="JPEG",
            append_to_existing_output=True
        )
        pose_counter += 1
    else:
        bproc.camera.remove_last_added_camera_pose()


