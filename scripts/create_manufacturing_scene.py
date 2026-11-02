
import blenderproc as bproc
import numpy as np
import time
import os

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

# Main extrusion body
create_box(
    name="horizontal_extrusion",
    location=[0, 0, 0.12],
    scale=[0.65, 0.04, 0.04],
    material=aluminium,
    category_id=2
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
# 9. CAMERA
# ============================================================

bproc.camera.set_resolution(640, 480)

camera_pose = bproc.math.build_transformation_mat(
    [1.5, -2.0, 1.4],
    [1.15, 0.0, 0.65]
)

bproc.camera.add_camera_pose(camera_pose)

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

os.makedirs("outputs/manufacturing_scene", exist_ok=True)

bproc.writer.write_hdf5(
    "outputs/manufacturing_scene",
    data
)

print(
    "\nRENDER TIME:",
    time.time() - start_time,
    "seconds"
)

print("\nImproved manufacturing scene completed!")
