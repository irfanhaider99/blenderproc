import blenderproc as bproc
import argparse
import os
import shutil
import json
import numpy as np
import re
import bpy

# ========== ARG PARSING ==========
parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path')
parser.add_argument('bop_dataset_name')
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures")
parser.add_argument('output_dir')
parser.add_argument('--N', type=int, default=4,
                    help="Number of object-sampling iterations")
args = parser.parse_args()

bproc.init()

# Set background to a dark gray color
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = [0.05, 0.05, 0.05, 1.0]  # RGB + Alpha
bg.inputs[1].default_value = 1.0  # Strength

# ========== LOAD CAMERA INTRINSICS ==========
# bproc.loader.load_bop_intrinsics(
#     bop_dataset_path=os.path.join(args.bop_parent_path, args.bop_dataset_name)
# )

#### ========== LOAD CAMERA INTRINSICS ==========
K = np.array([
    [2346.1826, 0.0,       956.1507],
    [0.0,       2351.0242, 613.65509],
    [0.0,       0.0,       1.0]
])

width = 1936
height = 1216

bproc.camera.set_resolution(width, height)
bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

#bproc.world.set_world_background([0.05, 0.05, 0.05, 1.0])
#bproc.world.set_world_background_color([0.05, 0.05, 0.05, 1.0])
#bproc.world.clear_light_sources()



# ========== BUILD ROOM ONCE ==========
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2,2,1]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[0,-2,2], rotation=[-1.570796,0,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[0, 2,2], rotation=[ 1.570796,0,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[2, 0,2], rotation=[0,-1.570796,0]),
    bproc.object.create_primitive('PLANE', scale=[2,2,1], location=[-2,0,2], rotation=[0, 1.570796,0])
]
for plane in room_planes:
    plane.enable_rigidbody(False,
                           collision_shape='BOX',
                           friction=1.0, # 1.0 is realistic, 100 is excessive
                           linear_damping=0.9,   #0.99
                           angular_damping=0.9) #0.99

# Emissive ceiling
light_plane = bproc.object.create_primitive('PLANE',
                                            scale=[3,3,1],
                                            location=[0,0,10])
light_plane.set_name("light_plane")

# Point light
light_point = bproc.types.Light()

# ========== ENABLE DEPTH OUTPUT ==========
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(32)

# ========== RENDER SETTINGS ==========
bproc.renderer.set_output_format(enable_transparency=False)

# ========== POSE SAMPLING FUNCTION ==========
def sample_pose(obj):
    loc = np.random.uniform([-0.15, -0.15, 0.05], [0.15, 0.15, 0.2])  # Raised z min to 0.05
    #loc = np.random.uniform([-0.15, -0.15, 0.5], [0.15, 0.15, 0.8])  # Higher Z
    obj.set_location(loc)
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
    
# ========== RENDER BUFFERS ==========
all_colors    = []
all_depths    = []
all_segment   = []
frame_cam_ids = []
pose_counter  = 0
ROTATIONS = 2

# ========== MAIN LOOP ==========
objs = []  # Initialize empty list to track objects
for iteration in range(args.N):
    # Clear camera poses
    bproc.camera.camera_poses = []
    # all_colors    = []
    # all_depths    = []
    
    # Delete objects from previous iteration
    # for o in objs:
    #     bproc.object.delete(o)
    objs = []  # Reset objs list
    
    # -- Randomize room texture --
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    chosen_tex = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(chosen_tex)

    # -- Randomize emissive plane --
    lm = bproc.material.create("bright_emission")
    lm.make_emissive(emission_strength=7.5, emission_color=[1.0, 1.0, 1.0, 1.0])
    light_plane.replace_materials(lm)


    # Set constant lighting instead of randomizing too much
    #light_point.set_energy(300)
    #light_point.set_color([1.0, 1.0, 1.0])
    #light_point.set_location([0, 0, 2])  # Place above the scene
    #light_plane.replace_materials(lm)
    light_point.set_energy(100)
    light_point.set_color([1.0, 1.0, 1.0])
    light_point.set_location([0, 0, 2])

    light_secondary = bproc.types.Light()
    light_secondary.set_type("POINT")
    light_secondary.set_location([1, -1, 1])
    light_secondary.set_energy(75)
    light_secondary.set_color([1.0, 1.0, 1.0])

    floor_light = bproc.types.Light()
    floor_light.set_type("POINT")
    floor_light.set_location([0, 0, 0.1])
    floor_light.set_energy(25)
    floor_light.set_color([1.0, 1.0, 1.0])


    """# -- Randomize point light --
    light_point.set_energy(np.random.uniform(150,400))
    light_point.set_color(np.random.uniform([0.8,0.8,0.8], [1.0,1.0,1.0]
    light_point.set_location(
        bproc.sampler.shell(center=[0,0,0],
                            radius_min=1, radius_max=1.5,
                            elevation_min=5, elevation_max=89,
                            uniform_volume=False)
    )"""

    # -- Reset transforms & load random BOP objects --
    bproc.utility.reset_keyframes()
    objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(args.bop_parent_path, args.bop_dataset_name),
        mm2m=True,
        sample_objects=True,
        num_of_objs_to_sample=8
    )

def sample_pose_with_distance(obj, placed_objs, min_dist=0.05, max_tries=100):
    tries = 0
    while tries < max_tries:
        # Sample a random location
        loc = np.random.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.25])
        
        # Check distance to already placed objects
        too_close = False
        for placed in placed_objs:
            if np.linalg.norm(loc - placed.get_location()) < min_dist:
                too_close = True
                break
        
        if not too_close:
            obj.set_location(loc)
            obj.set_rotation_euler(bproc.sampler.uniformSO3())
            return True
        
        tries += 1

    print(f"❌ Failed to place object {obj.get_name()} with sufficient spacing after {max_tries} tries.")
    return False


 # -- PBR randomization & physics setup --
for o in objs:
    o.enable_rigidbody(True,
                       friction=100.0,
                       linear_damping=0.99,
                       angular_damping=0.99)
    o.set_shading_mode('auto')
    mat = o.get_materials()[0]

    if o.get_cp("bop_dataset_name") in ['itodd', 'tless']:
        grey = np.random.uniform(0.3, 0.6)
        mat.set_principled_shader_value("Base Color", [grey, grey, grey, 1])
    else:
       base_color = np.random.uniform(0.4, 0.65, 3)
       mat.set_principled_shader_value("Base Color", base_color.tolist() + [1.0])
       #mat.set_principled_shader_value("Base Color", np.clip(np.random.uniform(0.2, 0.6, 3), 0.2, 0.6).tolist() + [1.0])

    try:
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.3, 0.5))
        mat.set_principled_shader_value("Specular", 0.4)
    except Exception as e:
        print(f"⚠️ Could not set shader values on material '{mat.get_name()}': {e}")



    # -- Drop & settle via physics --
    # -- Custom distance-aware sampling --
placed_objs = []
for o in objs:
    if sample_pose_with_distance(o, placed_objs, min_dist=0.0):
        placed_objs.append(o)

    #bproc.object.sample_poses(objs,
     #                         sample_pose_func=sample_pose,
      #                        max_tries=1000)
bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=3,
        max_simulation_time=5,
        check_object_interval=1,
        substeps_per_frame=10,
        solver_iters=10
)

# -- Filter valid objects (not sunk below floor) --
filtered_objs = []
for o in placed_objs:
    loc = o.get_location()
    if loc[2] < 0.03:
        loc[2] = 0.05
        o.set_location(loc)
    if loc[2] > 0.01:
        filtered_objs.append(o)

    objs = filtered_objs  
    if len(objs) == 0:
      print("⚠️ No valid objects found after simulation — skipping rendering.")
      continue
    if len(objs) < 1:
       print("⚠️ Too few objects — skipping.")
       continue
 


    # ========== FIX OBJECTS BELOW FLOOR ==========
for o in objs:
    loc = o.get_location()
    if loc[2] < 0.05:
        loc[2] = 0.05  # Lift object slightly to avoid penetrating floor
        o.set_location(loc)

    # NEW (less strict — keep all placed objects)
    objs = placed_objs
    print(f"✅ Proceeding with {len(objs)} objects for rendering.")
    if len(objs) == 0:
     print("⚠️ Still no objects found — skipping frame.")
     continue

    
    # -- Two POI camera views per iteration --
    cam_angles = 0
    ANGLES = 4
    # for cam_id in [1, 2]:
    while cam_angles < ANGLES:
        bvh = bproc.object.create_bvh_tree_multi_objects(objs)
        poi = bproc.object.compute_poi(objs)
        cam_loc = bproc.sampler.shell(center=poi,
                                      radius_min=0.4, radius_max=0.6,
                                      elevation_min=20, elevation_max=50,
                                      azimuth_min=-90, azimuth_max=90,   # full rotation
                                      uniform_volume=True)
        cam_rot = bproc.camera.rotation_from_forward_vec(
            poi - cam_loc,
            inplane_rot=np.random.uniform(-0.1,0.1)
        )
        cam_pose = bproc.math.build_transformation_mat(cam_loc, cam_rot)

        if not bproc.camera.perform_obstacle_in_view_check(cam_pose,
                                                           {"min":0.2},
                                                           bvh):
            print("⚠️ View check failed, skipping camera pose.")
            continue
        
        cam_angles += 1
        print(f"📸 Rendering {pose_counter + 1} with {len(objs)} objects.")

        # -- Rotate objects 10× & render one frame per rotation --
        for _ in range(ROTATIONS):
            axis = np.random.choice([0,1,2])
            for o in objs:
                eul = o.get_rotation_euler()
                eul[axis] += np.pi/45  # 36°
                o.set_rotation_euler(eul)
        for _ in range(ROTATIONS):
              axis = np.random.choice([0,1,2])
              for o in objs:
                 eul = o.get_rotation_euler()
                 eul[axis] += np.pi/45
                 o.set_rotation_euler(eul)
    
        bproc.camera.add_camera_pose(cam_pose)
        data = bproc.renderer.render()

        all_colors.append(data["colors"][-1])
        all_depths.append(data["depth"][-1])
        print(f"✅ Frame {pose_counter} rendered.")
        pose_counter += 1  # Increment here per frame


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