import blenderproc as bproc
import time

bproc.init()
bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type="CUDA")

bproc.object.create_primitive("CUBE")
light = bproc.types.Light()
light.set_location([2, -2, 4])
light.set_energy(500)

bproc.camera.set_resolution(640, 480)
bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([0, -4, 2], [1.1, 0, 0]))
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

t = time.time()
data = bproc.renderer.render()
print("RENDER TIME (s):", time.time() - t)
bproc.writer.write_hdf5("outputs/test", data)