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

# Main extrusion body: reserved category ID 4 for pose estimation
horizontal_extrusion = create_box(
    name="horizontal_extrusion",
    location=[0, 0, 0.12],
    scale=[0.65, 0.04, 0.04],
    material=aluminium,
    category_id=4
)

# Two grooves on the top face
for i, y in enumerate([-0.020, 0.020]):

    create_box(
        name=f"horizontal_top_slot_{i}",
        location=[0, y, 0.1615],
        scale=[0.65, 0.004, 0.0015],
        material=slot_material,
        category_id=2
    )

# One groove on the front face
create_box(
    name="horizontal_front_slot",
    location=[0, -0.0415, 0.12],
    scale=[0.65, 0.0015, 0.004],
    material=slot_material,
    category_id=2
)

# ============================================================
# 6. VERTICAL ALUMINIUM EXTRUSION
# ============================================================

create_box(
    name="vertical_extrusion",
    location=[0.35, 0, 0.52],
    scale=[0.04, 0.04, 0.40],
    material=aluminium,
    category_id=2
)

# Groove on the front face
create_box(
    name="vertical_front_slot",
    location=[0.35, -0.0415, 0.52],
    scale=[0.004, 0.0015, 0.40],
    material=slot_material,
    category_id=2
)

# Groove on the right-facing side
create_box(
    name="vertical_side_slot",
    location=[0.3915, 0, 0.52],
    scale=[0.0015, 0.004, 0.40],
    material=slot_material,
    category_id=2
)

# ============================================================
# 7. L-SHAPED MOUNTING BRACKETS
# ============================================================

# Bracket on the front side of the assembly

create_box(
    name="front_bracket_horizontal",
    location=[0.30, -0.055, 0.18],
    scale=[0.12, 0.008, 0.025],
    material=steel,
    category_id=3
)

create_box(
    name="front_bracket_vertical",
    location=[0.405, -0.055, 0.255],
    scale=[0.008, 0.008, 0.075],
    material=steel,
    category_id=3
)

# Bracket on the rear side of the assembly

create_box(
    name="rear_bracket_horizontal",
    location=[0.30, 0.055, 0.18],
    scale=[0.12, 0.008, 0.025],
    material=steel,
    category_id=3
)

create_box(
    name="rear_bracket_vertical",
    location=[0.405, 0.055, 0.255],
    scale=[0.008, 0.008, 0.075],
    material=steel,
    category_id=3
)

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

STEREO_BASELINE_M = 0.095

T_plus = np.eye(4)
T_plus[0, 3] = STEREO_BASELINE_M

T_minus = np.eye(4)
T_minus[0, 3] = -STEREO_BASELINE_M

camera_pose_plus = camera_pose @ T_plus
camera_pose_minus = camera_pose @ T_minus

camera_poses = {
    "original": camera_pose,
    "plus_x": camera_pose_plus,
    "minus_x": camera_pose_minus,
}

# Register in the exact order used by the metadata.
for view_name, pose in camera_poses.items():
    bproc.camera.add_camera_pose(pose)
    print(f"Registered camera view: {view_name}")
    print(pose)

print("\nCamera location:", camera_location)
print("Camera look-at target:", look_at)
print("Camera forward vector:", forward_vec)
print("Camera rotation matrix:")
print(rotation_matrix)

print("\nCamera pose:")
print(camera_pose)

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

STEREO_OUTPUT_DIR = "outputs/stereo_geometry"

os.makedirs(STEREO_OUTPUT_DIR, exist_ok=True)

bproc.writer.write_hdf5(
    STEREO_OUTPUT_DIR,
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

metadata = {
    "camera": {
        "model": "Intel RealSense D455 (synthetic approximation)",
        "width": WIDTH,
        "height": HEIGHT,
        "intrinsics": K.tolist(),

        # Original camera pose retained for compatibility.
        "camera_to_world_blender": camera_pose.tolist(),

        # Explicit poses for the three rendered views.
        "camera_poses": {
            name: pose.tolist()
            for name, pose in camera_poses.items()
        },

        "stereo_geometry": {
            "nominal_baseline_m": STEREO_BASELINE_M,
            "offset_axis": "local_camera_x",
            "view_order": [
                "original",
                "plus_x",
                "minus_x"
            ],
            "note": (
                "Geometric sensitivity audit using virtual camera "
                "displacements; not a calibrated reconstruction "
                "of the physical D455 stereo pair."
            )
        },

        "horizontal_fov_deg": HFOV_DEG,
        "vertical_fov_deg": VFOV_DEG,
        "physical_calibration_performed": False
    },
    "target": {
        "name": "horizontal_extrusion",
        "category_id": 4,
        "object_to_world": np.asarray(
            horizontal_extrusion.get_local2world_mat()
        ).tolist(),
        "geometry": "rectangular extrusion body",
        "half_extents_m": [0.65, 0.04, 0.04]
    }
}

metadata_path = os.path.join(
    STEREO_OUTPUT_DIR,
    "ground_truth.json"
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nGround-truth metadata saved to:", metadata_path)