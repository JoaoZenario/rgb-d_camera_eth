import h5py
import json
import numpy as np
import open3d as o3d

with open("outputs/manufacturing_scene/ground_truth.json") as f:
    meta = json.load(f)

K = np.asarray(meta["camera"]["intrinsics"], dtype=float)
T_cw = np.asarray(
    meta["camera"]["camera_to_world_blender"],
    dtype=float
)
T_gt = np.asarray(
    meta["target"]["object_to_world"],
    dtype=float
)

half = np.asarray(meta["target"]["half_extents_m"])

with h5py.File("outputs/manufacturing_scene/0.hdf5", "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    seg = np.squeeze(f["category_id_segmaps"][...])

mask = (
    (seg == 4)
    & np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

v, u = np.where(mask)
z = depth[v, u]

points_cv = np.column_stack((
    (u - K[0, 2]) * z / K[0, 0],
    (v - K[1, 2]) * z / K[1, 1],
    z
))

S = np.diag([1.0, -1.0, -1.0])
points_cam = (S @ points_cv.T).T

points_world = (
    T_cw[:3, :3] @ points_cam.T
).T + T_cw[:3, 3]

gt_center = T_gt[:3, 3]

print("\n========== OBSERVED GEOMETRY ==========")
print("Point count:", len(points_world))
print("Ground-truth centre:", gt_center)
print("Observed centroid:", points_world.mean(axis=0))
print("Observed minimum:", points_world.min(axis=0))
print("Observed maximum:", points_world.max(axis=0))
print("Observed extent:", np.ptp(points_world, axis=0))

print("\n========== AXIS-ALIGNED DISTRIBUTION ==========")

for axis, name in enumerate(["X", "Y", "Z"]):
    values = points_world[:, axis]
    print(
        f"{name}: "
        f"P05={np.percentile(values, 5):.4f} m, "
        f"P25={np.percentile(values, 25):.4f} m, "
        f"P50={np.percentile(values, 50):.4f} m, "
        f"P75={np.percentile(values, 75):.4f} m, "
        f"P95={np.percentile(values, 95):.4f} m"
    )

print("\n========== DISTANCE FROM TRUE CENTRE ==========")

distances = np.linalg.norm(points_world - gt_center, axis=1)

print("Mean:", distances.mean())
print("Median:", np.median(distances))
print("P95:", np.percentile(distances, 95))

# Save point cloud for visual inspection.
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points_world)

o3d.io.write_point_cloud(
    "outputs/pca_icp/observed_geometry.ply",
    cloud
)

print("\nSaved: outputs/pca_icp/observed_geometry.ply")