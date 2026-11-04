import json
import h5py
import numpy as np

from pathlib import Path

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/pca_obb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rotation_error_degrees(R_est, R_gt):
    R_error = R_est @ R_gt.T
    cosine = np.clip(
        (np.trace(R_error) - 1.0) / 2.0,
        -1.0,
        1.0
    )
    return np.degrees(np.arccos(cosine))


# ------------------------------------------------------------
# LOAD GROUND TRUTH AND CAMERA DATA
# ------------------------------------------------------------

with open(META_PATH) as f:
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

half_extents = np.asarray(
    meta["target"]["half_extents_m"],
    dtype=float
)

R_gt = T_gt[:3, :3]
t_gt = T_gt[:3, 3]


# ------------------------------------------------------------
# LOAD SEGMENTED DEPTH
# ------------------------------------------------------------

with h5py.File(HDF5_PATH, "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

mask = (
    (segmentation == 4)
    & np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

v, u = np.where(mask)
z = depth[v, u]

print("Target pixels:", len(z))


# ------------------------------------------------------------
# BACK-PROJECT INTO WORLD COORDINATES
# ------------------------------------------------------------

points_cv = np.column_stack((
    (u - K[0, 2]) * z / K[0, 0],
    (v - K[1, 2]) * z / K[1, 1],
    z
))

S = np.diag([1.0, -1.0, -1.0])

points_camera_blender = (
    S @ points_cv.T
).T

points_world = (
    T_cw[:3, :3] @ points_camera_blender.T
).T + T_cw[:3, 3]


# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

sample_mean = points_world.mean(axis=0)
centered = points_world - sample_mean

covariance = np.cov(centered.T)

eigenvalues, eigenvectors = np.linalg.eigh(covariance)

order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
R_pca = eigenvectors[:, order]

# Enforce a right-handed coordinate frame.
if np.linalg.det(R_pca) < 0:
    R_pca[:, 2] *= -1


# ------------------------------------------------------------
# PROJECT INTO PCA COORDINATES
# ------------------------------------------------------------

local_points = centered @ R_pca

local_min = local_points.min(axis=0)
local_max = local_points.max(axis=0)

local_midpoint = (local_min + local_max) / 2.0

estimated_center = (
    sample_mean + R_pca @ local_midpoint
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

translation_error = np.linalg.norm(
    estimated_center - t_gt
)

# PCA axis signs and ordering are ambiguous.
# For this first test, compare the long axis as an undirected axis.
long_axis_est = R_pca[:, 0]
long_axis_gt = R_gt[:, 0]

axis_error = np.degrees(
    np.arccos(
        np.clip(
            abs(np.dot(long_axis_est, long_axis_gt)),
            -1.0,
            1.0
        )
    )
)

estimated_extents = local_max - local_min
expected_extents = 2.0 * half_extents

print("\n========== PCA-OBB RESULTS ==========")

print("PCA eigenvalues:", eigenvalues)

print("Estimated centre:", estimated_center)
print("Ground-truth centre:", t_gt)

print(
    "Translation error:",
    translation_error * 1000,
    "mm"
)

print("Long-axis error:", axis_error, "degrees")

print("\nEstimated extents:", estimated_extents)
print("Expected extents:", expected_extents)

print("\nPCA rotation matrix:\n", R_pca)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = {
    "translation_error_mm": float(
        translation_error * 1000
    ),
    "long_axis_error_deg": float(axis_error),
    "estimated_center": estimated_center.tolist(),
    "ground_truth_center": t_gt.tolist(),
    "estimated_extents": estimated_extents.tolist(),
    "expected_extents": expected_extents.tolist(),
    "pca_eigenvalues": eigenvalues.tolist(),
    "pca_rotation": R_pca.tolist(),
    "physical_calibration_performed": False
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved:", OUTPUT_DIR / "results.json")