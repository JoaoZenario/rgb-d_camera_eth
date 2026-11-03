
import json
import h5py
import numpy as np
import open3d as o3d

from scipy.spatial import cKDTree

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

with open(META_PATH) as f:
    meta = json.load(f)

K = np.asarray(meta["camera"]["intrinsics"])
T_cw = np.asarray(meta["camera"]["camera_to_world_blender"])
T_gt = np.asarray(meta["target"]["object_to_world"])
half = np.asarray(meta["target"]["half_extents_m"])

with h5py.File(HDF5_PATH, "r") as f:
    depth = np.asarray(f["depth"])
    seg = np.asarray(f["category_id_segmaps"])

mask = (
    (seg == 4)
    & np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

v, u = np.where(mask)
z = depth[v, u]

# Back-project in OpenCV camera coordinates.
points_cv = np.column_stack((
    (u - K[0, 2]) * z / K[0, 0],
    (v - K[1, 2]) * z / K[1, 1],
    z
))

# Convert OpenCV camera axes to Blender camera axes.
S = np.diag([1.0, -1.0, -1.0])
points_cam_blender = (S @ points_cv.T).T

# Transform into world coordinates.
points_world = (
    T_cw[:3, :3] @ points_cam_blender.T
).T + T_cw[:3, 3]

observed = o3d.geometry.PointCloud()
observed.points = o3d.utility.Vector3dVector(points_world)

# Build the exact rectangular template.
mesh = o3d.geometry.TriangleMesh.create_box(
    width=2 * half[0],
    height=2 * half[1],
    depth=2 * half[2]
)
mesh.translate(-half)

template = mesh.sample_points_uniformly(50000)

# Transform template with ground truth.
template_gt = o3d.geometry.PointCloud(template)
template_gt.transform(T_gt)

# Compare nearest-neighbour distances.
obs_points = np.asarray(observed.points)
gt_points = np.asarray(template_gt.points)

tree_gt = cKDTree(gt_points)
d_obs_to_gt, _ = tree_gt.query(obs_points, k=1)

tree_obs = cKDTree(obs_points)
d_gt_to_obs, _ = tree_obs.query(gt_points, k=1)

print("\n========== GROUND-TRUTH ALIGNMENT ==========")
print("Observed points:", len(obs_points))
print("Template points:", len(gt_points))

print("\nObserved -> ground-truth template:")
print("Mean distance (mm):", np.mean(d_obs_to_gt) * 1000)
print("Median distance (mm):", np.median(d_obs_to_gt) * 1000)
print("95th percentile (mm):", np.percentile(d_obs_to_gt, 95) * 1000)
print("Within 5 mm:", np.mean(d_obs_to_gt < 0.005))

print("\nGround-truth template -> observed:")
print("Mean distance (mm):", np.mean(d_gt_to_obs) * 1000)
print("Median distance (mm):", np.median(d_gt_to_obs) * 1000)
print("95th percentile (mm):", np.percentile(d_gt_to_obs, 95) * 1000)
print("Within 5 mm:", np.mean(d_gt_to_obs < 0.005))

# Check target centre in camera coordinates.
target_center_world = T_gt[:3, 3]
target_center_cam_blender = (
    T_cw[:3, :3].T
    @ (target_center_world - T_cw[:3, 3])
)

target_center_cv = S @ target_center_cam_blender

u_center = (
    K[0, 0] * target_center_cv[0] / target_center_cv[2]
    + K[0, 2]
)
v_center = (
    K[1, 1] * target_center_cv[1] / target_center_cv[2]
    + K[1, 2]
)

print("\n========== CAMERA PROJECTION CHECK ==========")
print("Target centre in Blender camera frame:",
      target_center_cam_blender)
print("Target centre in OpenCV camera frame:",
      target_center_cv)
print("Projected target centre (u, v):",
      (u_center, v_center))
print("Observed target pixel centroid:",
      (np.mean(u), np.mean(v)))

# ICP with the correct pose as initialization.
source = template.voxel_down_sample(0.003)
target = observed.voxel_down_sample(0.003)

source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015, max_nn=30
    )
)
target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015, max_nn=30
    )
)

result = o3d.pipelines.registration.registration_icp(
    source,
    target,
    0.02,
    T_gt,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("\n========== ICP FROM GROUND TRUTH ==========")
print("Fitness:", result.fitness)
print("Inlier RMSE (mm):", result.inlier_rmse * 1000)
print("Result transformation:\n", result.transformation)

o3d.io.write_point_cloud(
    "outputs/pose_estimation_test/observed_world.ply",
    observed
)

o3d.io.write_point_cloud(
    "outputs/pose_estimation_test/template_ground_truth.ply",
    template_gt
)

print("\nDiagnostic point clouds saved.")
