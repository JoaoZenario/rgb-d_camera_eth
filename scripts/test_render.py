
import blenderproc as bproc
import time
import numpy as np

# 1. Initialise BlenderProc
bproc.init()

bproc.renderer.set_render_devices(
    use_only_cpu=False,
    desired_gpu_device_type="CUDA"
)

# 2. Create a simple test scene
bproc.object.create_primitive("CUBE")

light = bproc.types.Light()
light.set_location([2, -2, 4])
light.set_energy(500)

# 3. Configure the virtual camera
bproc.camera.set_resolution(640, 480)

bproc.camera.add_camera_pose(
    bproc.math.build_transformation_mat(
        [0, -4, 2],
        [1.1, 0, 0]
    )
)

# 4. Configure rendering
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

# 5. Render the scene
t = time.time()
data = bproc.renderer.render()

print("\nOUTPUT KEYS:", data.keys())

# 6. Inspect RGB and depth outputs
for key, frames in data.items():
    print(f"\n{key}:")
    print("  Type:", type(frames))
    print("  Number of frames:", len(frames))

    for i, frame in enumerate(frames):
        array = np.asarray(frame)

        print(f"  Frame {i}:")
        print("    Shape:", array.shape)
        print("    Data type:", array.dtype)
        print("    Min:", np.nanmin(array))
        print("    Max:", np.nanmax(array))

# 7. Inspect valid depth values
depth = np.asarray(data["depth"][0])

# Exclude invalid/background depth values
valid = np.isfinite(depth) & (depth < 1000)

print("\nVALID DEPTH STATISTICS")
print("  Valid pixels:", np.count_nonzero(valid))
print("  Invalid pixels:", np.count_nonzero(~valid))

if np.any(valid):
    print("  Minimum:", depth[valid].min(), "m")
    print("  Maximum:", depth[valid].max(), "m")
    print("  Mean:", depth[valid].mean(), "m")
    print("  Valid pixel percentage:", 100 * valid.mean(), "%")

# 8. Save rendered data
bproc.writer.write_hdf5("outputs/test", data)

print("\nRENDER TIME (s):", time.time() - t)
print("Rendering and saving completed.")
