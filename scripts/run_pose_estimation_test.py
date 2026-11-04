import json
import h5py
import numpy as np
import open3d as o3d

from pathlib import Path
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
METADATA_PATH = "outputs/manufacturing_scene/ground_truth.json"
OUTPUT_DIR = Path("outputs/pose_estimation_test")

VOXEL_SIZE = 0.005
NORMAL_RADIUS = 0.020
FEATURE_RADIUS = 0.040
RANSAC_DISTANCE = 0.015
ICP_DISTANCE = 0.015

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA AND GROUND TRUTH
# ============================================================

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

K = np.asarray(metadata["camera"]["intrinsics"], dtype=float)
camera_to_world_blender = np.asarray(
    metadata["camera"]["camera_to_world_blender"],
    dtype=float
)
object_to_world_gt = np.asarray(
    metadata["target"]["object_to_world"],
    dtype=float
)

half_extents = np.asarray(
    metadata["target"]["half_extents_m"],
    dtype=float
)

with h5py.File(HDF5_PATH, "r") as f:
    depth = np.asarray(f["depth"], dtype=np.float32)
    segmentation = np.asarray(f["category_id_segmaps"])

height, width = depth.shape

target_mask = segmentation == 4
valid_mask = (
    target_mask
    & np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

print("\n=== INPUT DATA ===")
print("Image resolution:", width, "x", height)
print("Target pixels:", int(target_mask.sum()))
print("Valid target depth pixels:", int(valid_mask.sum()))

if valid_mask.sum() < 100:
    raise RuntimeError("Too few valid target pixels for registration.")


# ============================================================
# BACK-PROJECT DEPTH INTO CAMERA COORDINATES
# ============================================================

v, u = np.where(valid_mask)
z = depth[v, u]

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

x = (u - cx) * z / fx
y = (v - cy) * z / fy

# OpenCV camera coordinates:
# X right, Y down, Z forward.
points_cv = np.column_stack((x, y, z))

# Blender camera coordinates:
# X right, Y up, Z backward.
#
# This conversion has determinant +1 and preserves
# a right-handed coordinate system.
opencv_to_blender_camera = np.diag([1.0, -1.0, -1.0])

points_blender_camera = (
    opencv_to_blender_camera @ points_cv.T
).T

# Convert camera-frame points into Blender world coordinates.
points_world = (
    camera_to_world_blender[:3, :3]
    @ points_blender_camera.T
).T + camera_to_world_blender[:3, 3]

observed = o3d.geometry.PointCloud()
observed.points = o3d.utility.Vector3dVector(points_world)

print("Observed point cloud:", len(observed.points), "points")


# ============================================================
# BUILD KNOWN RECTANGULAR TEMPLATE IN OBJECT-LOCAL FRAME
# ============================================================

# BlenderProc cube scale values are half-extents.
template_mesh = o3d.geometry.TriangleMesh.create_box(
    width=2.0 * half_extents[0],
    height=2.0 * half_extents[1],
    depth=2.0 * half_extents[2]
)

# Open3D create_box starts at the origin.
# Centre it so that its local origin matches the object origin.
template_mesh.translate(-half_extents)

template_mesh.compute_vertex_normals()

template = template_mesh.sample_points_uniformly(
    number_of_points=30000
)

print("Template point cloud:", len(template.points), "points")


# ============================================================
# DOWNSAMPLE AND ESTIMATE NORMALS
# ============================================================

source = template.voxel_down_sample(VOXEL_SIZE)
target = observed.voxel_down_sample(VOXEL_SIZE)

source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS,
        max_nn=30
    )
)

target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS,
        max_nn=30
    )
)

print("Downsampled template:", len(source.points))
print("Downsampled observation:", len(target.points))


# ============================================================
# COMPUTE FPFH FEATURES
# ============================================================

source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source,
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=FEATURE_RADIUS,
        max_nn=100
    )
)

target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target,
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=FEATURE_RADIUS,
        max_nn=100
    )
)


# ============================================================
# GLOBAL REGISTRATION: RANSAC
# ============================================================

print("\nRunning global RANSAC registration...")

distance_threshold = RANSAC_DISTANCE

ransac_result = (
    o3d.pipelines.registration
    .registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=(
            o3d.pipelines.registration
            .TransformationEstimationPointToPoint(False)
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration
            .CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration
            .CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            )
        ],
        criteria=(
            o3d.pipelines.registration
            .RANSACConvergenceCriteria(100000, 0.999)
        )
    )
)

print("\nRANSAC fitness:", ransac_result.fitness)
print("RANSAC inlier RMSE:", ransac_result.inlier_rmse)
print("RANSAC transformation:\n", ransac_result.transformation)


# ============================================================
# ICP REFINEMENT
# ============================================================

print("\nRunning point-to-plane ICP...")

icp_result = (
    o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_DISTANCE,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
)

estimated_pose = icp_result.transformation

print("\nICP fitness:", icp_result.fitness)
print("ICP inlier RMSE:", icp_result.inlier_rmse)
print("Estimated object-to-world pose:\n", estimated_pose)


# ============================================================
# POSE ERRORS
# ============================================================

R_est = estimated_pose[:3, :3]
R_gt = object_to_world_gt[:3, :3]

t_est = estimated_pose[:3, 3]
t_gt = object_to_world_gt[:3, 3]

translation_error = np.linalg.norm(t_est - t_gt)

R_error = R_est @ R_gt.T
cos_angle = np.clip(
    (np.trace(R_error) - 1.0) / 2.0,
    -1.0,
    1.0
)

rotation_error_deg = np.degrees(np.arccos(cos_angle))


# ============================================================
# ADD-S: SYMMETRY-AWARE MODEL DISTANCE
# ============================================================

template_local_points = np.asarray(template.points)

estimated_points = (
    R_est @ template_local_points.T
).T + t_est

ground_truth_points = (
    R_gt @ template_local_points.T
).T + t_gt

# ADD-S uses nearest-neighbour distances, making it suitable
# for geometrically symmetric objects.
tree = cKDTree(ground_truth_points)

distances, _ = tree.query(
    estimated_points,
    k=1
)

adds_error = float(np.mean(distances))


# ============================================================
# SIMULATED GRASP-FEASIBILITY CRITERION
# ============================================================

POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE_DEG = 5.0

grasp_feasible = (
    translation_error <= POSITION_TOLERANCE
    and rotation_error_deg <= ORIENTATION_TOLERANCE_DEG
)

print("\n========== POSE RESULTS ==========")
print(f"Translation error: {translation_error * 1000:.3f} mm")
print(f"Rotation error: {rotation_error_deg:.3f} degrees")
print(f"ADD-S: {adds_error * 1000:.3f} mm")
print(f"ICP fitness: {icp_result.fitness:.4f}")
print(f"ICP inlier RMSE: {icp_result.inlier_rmse * 1000:.3f} mm")
print("Simulated grasp-feasibility:", grasp_feasible)


# ============================================================
# SAVE RESULTS AND POINT CLOUDS
# ============================================================

o3d.io.write_point_cloud(
    str(OUTPUT_DIR / "observed_target.ply"),
    observed
)

aligned_template = o3d.geometry.PointCloud(template)
aligned_template.transform(estimated_pose)

o3d.io.write_point_cloud(
    str(OUTPUT_DIR / "estimated_template.ply"),
    aligned_template
)

o3d.io.write_point_cloud(
    str(OUTPUT_DIR / "ground_truth_template.ply"),
    o3d.geometry.PointCloud(template).transform(object_to_world_gt)
)

results = {
    "translation_error_m": float(translation_error),
    "rotation_error_deg": float(rotation_error_deg),
    "adds_error_m": adds_error,
    "ransac_fitness": float(ransac_result.fitness),
    "ransac_inlier_rmse_m": float(ransac_result.inlier_rmse),
    "icp_fitness": float(icp_result.fitness),
    "icp_inlier_rmse_m": float(icp_result.inlier_rmse),
    "simulated_grasp_feasible": bool(grasp_feasible),
    "position_tolerance_m": POSITION_TOLERANCE,
    "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
    "target_pixels": int(target_mask.sum()),
    "valid_target_depth_pixels": int(valid_mask.sum())
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:", OUTPUT_DIR)