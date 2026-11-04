import blenderproc as bproc
import numpy as np
import time
import os
import json

# ============================================================
# 1. INITIALISE BLENDERPROC
# ============================================================

bproc.init()

bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type="CUDA"
)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def create_material(name, color, metallic=0.0, roughness=0.5):

    material = bproc.material.create(name)

    material.set_principled_shader_value(
        "Base Color", color
    )

    material.set_principled_shader_value(
        "Metallic", metallic
    )

    material.set_principled_shader_value(
        "Roughness", roughness
    )

    return material


def create_box(name, location, scale, material, category_id):

    obj = bproc.object.create_primitive(
        "CUBE",
        scale=scale,
        location=location
    )

    obj.set_name(name)
    obj.replace_materials(material)
    obj.set_cp("category_id", category_id)

    return obj


# ============================================================
# 3. MATERIALS
# ============================================================

aluminium = create_material(
    "Aluminium",
    [0.65, 0.68, 0.72, 1.0],
    metallic=0.8,
    roughness=0.3
)

slot_material = create_material(
    "Dark_T_Slot",
    [0.055, 0.065, 0.075, 1.0],
    metallic=0.5,
    roughness=0.45
)

steel = create_material(
    "Steel",
    [0.35, 0.38, 0.42, 1.0],
    metallic=0.75,
    roughness=0.35
)

table_material = create_material(
    "Workbench",
    [0.18, 0.20, 0.23, 1.0],
    metallic=0.15,
    roughness=0.65
)

# ============================================================
# 4. WORKBENCH
# ============================================================

create_box(
    name="workbench",
    location=[0, 0, 0],
    scale=[1.2, 0.8, 0.05],
    material=table_material,
    category_id=1
)

# ============================================================
# 5. HORIZONTAL ALUMINIUM EXTRUSION
# ============================================================

assembly_parts = []

# Main extrusion body: reserved category ID 4 for pose estimation
horizontal_extrusion = create_box(
    name="horizontal_extrusion",
    location=[0, 0, 0.12],
    scale=[0.65, 0.04, 0.04],
    material=aluminium,
    category_id=4
)

assembly_parts.append(horizontal_extrusion)

# Two grooves on the top face
for i, y in enumerate([-0.020, 0.020]):

    slot = create_box(
        name=f"horizontal_top_slot_{i}",
        location=[0, y, 0.1615],
        scale=[0.65, 0.004, 0.0015],
        material=slot_material,
        category_id=2
    )
    assembly_parts.append(slot)

# One groove on the front face
horizontal_front_slot = create_box(
    name="horizontal_front_slot",
    location=[0, -0.0415, 0.12],
    scale=[0.65, 0.0015, 0.004],
    material=slot_material,
    category_id=2
)
assembly_parts.append(horizontal_front_slot)

# ============================================================
# 6. VERTICAL ALUMINIUM EXTRUSION
# ============================================================

vertical_extrusion = create_box(
    name="vertical_extrusion",
    location=[0.35, 0, 0.52],
    scale=[0.04, 0.04, 0.40],
    material=aluminium,
    category_id=2
)
assembly_parts.append(vertical_extrusion)

# Groove on the front face
vertical_front_slot = create_box(
        name="vertical_front_slot",
        location=[0.35, -0.0415, 0.52],
        scale=[0.004, 0.0015, 0.40],
        material=slot_material,
        category_id=2
    )
assembly_parts.append(vertical_front_slot)

# Groove on the right-facing side
vertical_side_slot = create_box(
    name="vertical_side_slot",
    location=[0.3915, 0, 0.52],
    scale=[0.0015, 0.004, 0.40],
    material=slot_material,
    category_id=2
)
assembly_parts.append(vertical_side_slot)

# ============================================================
# 7. L-SHAPED MOUNTING BRACKETS
# ============================================================

# Bracket on the front side of the assembly

front_bracket_horizontal = create_box(
    name="front_bracket_horizontal",
    location=[0.30, -0.055, 0.18],
    scale=[0.12, 0.008, 0.025],
    material=steel,
    category_id=3
)
assembly_parts.append(front_bracket_horizontal)

front_bracket_vertical = create_box(
    name="front_bracket_vertical",
    location=[0.405, -0.055, 0.255],
    scale=[0.008, 0.008, 0.075],
    material=steel,
    category_id=3
)
assembly_parts.append(front_bracket_vertical)

# Bracket on the rear side of the assembly

rear_bracket_horizontal = create_box(
    name="rear_bracket_horizontal",
    location=[0.30, 0.055, 0.18],
    scale=[0.12, 0.008, 0.025],
    material=steel,
    category_id=3
)
assembly_parts.append(rear_bracket_horizontal)

rear_bracket_vertical = create_box(
    name="rear_bracket_vertical",
    location=[0.405, 0.055, 0.255],
    scale=[0.008, 0.008, 0.075],
    material=steel,
    category_id=3
)
assembly_parts.append(rear_bracket_vertical)

# ============================================================
# RIGID ASSEMBLY PARENT
# ============================================================

assembly_root = bproc.object.create_empty("assembly_root")

assembly_root.set_location([0.0, 0.0, 0.0])
assembly_root.set_rotation_euler([0.0, 0.0, 0.0])

for part in assembly_parts:
    part.set_parent(assembly_root)

# Save the target's initial world pose before animation.
target_local_pose = np.asarray(
    horizontal_extrusion.get_local2world_mat()
).copy()

print("Assembly components parented:", len(assembly_parts))
print("Target initial pose:")
print(target_local_pose)

# ============================================================
# 8. LIGHTING
# ============================================================

light = bproc.types.Light()

light.set_type("AREA")
light.set_location([0, -1.5, 2.0])
light.set_energy(250)
light.set_color([1.0, 1.0, 1.0])

# ============================================================
# 9. CAMERA — PROVISIONAL D455-INFORMED INTRINSICS
# ============================================================

WIDTH = 640
HEIGHT = 360

# Approximate D455 depth field of view:
# horizontal: 86 degrees
# vertical:   57 degrees
#
# These are specification-informed synthetic intrinsics,
# NOT a physical calibration of a particular D455 unit.

HFOV_DEG = 86.0
VFOV_DEG = 57.0

fx = WIDTH / (2.0 * np.tan(np.deg2rad(HFOV_DEG / 2.0)))
fy = HEIGHT / (2.0 * np.tan(np.deg2rad(VFOV_DEG / 2.0)))

cx = (WIDTH - 1) / 2.0
cy = (HEIGHT - 1) / 2.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])

bproc.camera.set_resolution(WIDTH, HEIGHT)
bproc.camera.set_intrinsics_from_K_matrix(
    K,
    WIDTH,
    HEIGHT
)

print("\nCamera intrinsic matrix:")
print(K)

camera_location = np.array([0.85, -1.25, 0.85])
look_at = np.array([0.0, 0.0, 0.20])

# BlenderProc camera convention: local camera forward is -Z.
forward_vec = look_at - camera_location

rotation_matrix = bproc.camera.rotation_from_forward_vec(
    forward_vec
)

camera_pose = bproc.math.build_transformation_mat(
    camera_location,
    rotation_matrix
)

# ------------------------------------------------------------
# Three-view geometric stereo audit
# Frame 0: original camera
# Frame 1: camera displaced +95 mm along local camera X
# Frame 2: camera displaced -95 mm along local camera X
# ------------------------------------------------------------

# ============================================================
# RANDOMIZED POSE DATASET
# ============================================================

NUM_POSES = 200
DATASET_SEED = 20261104

rng = np.random.default_rng(DATASET_SEED)

pose_samples = []

# Remove previous keyframes before creating the dataset.
bproc.utility.reset_keyframes()

for frame_idx in range(NUM_POSES):

    yaw = rng.uniform(-np.pi, np.pi)

    dx = rng.uniform(-0.06, 0.06)
    dy = rng.uniform(-0.06, 0.06)

    # Parent transform: yaw about world Z and XY translation.
    c = np.cos(yaw)
    s = np.sin(yaw)

    T_parent = np.eye(4)
    T_parent[:3, :3] = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    T_parent[:3, 3] = [dx, dy, 0.0]

    # Animate the complete assembly.
    assembly_root.set_location(
        [dx, dy, 0.0],
        frame=frame_idx
    )

    assembly_root.set_rotation_euler(
        [0.0, 0.0, yaw],
        frame=frame_idx
    )

    # Fixed camera, explicitly assigned to this frame.
    bproc.camera.add_camera_pose(
        camera_pose,
        frame=frame_idx
    )

    # Ground-truth pose of category-4 extrusion.
    T_target_world = T_parent @ target_local_pose

    pose_samples.append({
        "sample_id": frame_idx,
        "frame": frame_idx,
        "yaw_rad": float(yaw),
        "translation_xy_m": [float(dx), float(dy)],
        "assembly_to_world": T_parent.tolist(),
        "target_to_world": T_target_world.tolist(),
        "camera_to_world_blender": camera_pose.tolist()
    })

print(f"Registered {NUM_POSES} randomized poses.")

# ============================================================
# 10. RENDER SETTINGS
# ============================================================

bproc.renderer.set_max_amount_of_samples(32)

bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

bproc.renderer.enable_segmentation_output(
    map_by=["category_id"],
    default_values={"category_id": 0}
)

# ============================================================
# 11. RENDER
# ============================================================

print("\nRendering improved manufacturing scene...")

start_time = time.time()

data = bproc.renderer.render()

print("\nOUTPUT KEYS:", data.keys())

# ============================================================
# 12. INSPECT OUTPUTS
# ============================================================

for key, frames in data.items():

    print(f"\n{key}:")

    for i, frame in enumerate(frames):

        array = np.asarray(frame)

        print(f"  Frame {i}:")
        print("    Shape:", array.shape)
        print("    Data type:", array.dtype)

        if np.issubdtype(array.dtype, np.number):
            print("    Min:", np.nanmin(array))
            print("    Max:", np.nanmax(array))

# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

DATASET_DIR = "outputs/randomized_pose_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

bproc.writer.write_hdf5(
    DATASET_DIR,
    data
)

print(
    "\nRENDER TIME:",
    time.time() - start_time,
    "seconds"
)

print("\nImproved manufacturing scene completed!")

# ============================================================
# 14. SAVE GROUND-TRUTH METADATA
# ============================================================

# ============================================================
# SAVE PER-SAMPLE GROUND TRUTH
# ============================================================

for i, sample in enumerate(pose_samples):

    seg = np.asarray(
        data["category_id_segmaps"][i]
    ).squeeze()

    depth_frame = np.asarray(
        data["depth"][i]
    ).squeeze()

    target_pixels = int(np.count_nonzero(seg == 4))

    valid_target_pixels = int(np.count_nonzero(
        (seg == 4)
        & np.isfinite(depth_frame)
        & (depth_frame > 0)
        & (depth_frame < 1000)
    ))

    sample["target_pixels"] = target_pixels
    sample["valid_target_depth_pixels"] = valid_target_pixels

    sample["target_depth_valid_fraction"] = (
        valid_target_pixels / target_pixels
        if target_pixels > 0 else 0.0
    )

metadata = {
    "dataset_name": "randomized_manufacturing_pose_dataset",
    "num_samples": NUM_POSES,
    "random_seed": DATASET_SEED,
    "camera_model": "Intel RealSense D455 synthetic approximation",
    "physical_calibration_performed": False,
    "image_width": WIDTH,
    "image_height": HEIGHT,
    "intrinsics": K.tolist(),
    "camera_to_world_blender": camera_pose.tolist(),
    "pose_distribution": {
        "yaw_rad": [-float(np.pi), float(np.pi)],
        "translation_x_m": [-0.06, 0.06],
        "translation_y_m": [-0.06, 0.06],
        "translation_z_m": [0.0, 0.0]
    },
    "samples": pose_samples
}

with open(
    os.path.join(DATASET_DIR, "pose_labels.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("Saved pose labels:", os.path.join(
    DATASET_DIR, "pose_labels.json"
))