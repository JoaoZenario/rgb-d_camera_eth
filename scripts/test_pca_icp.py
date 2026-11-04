import json
import h5py
import itertools
import numpy as np
import open3d as o3d

from pathlib import Path

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/pca_icp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.003
ICP_DISTANCE = 0.020

# ------------------------------------------------------------
# LOAD DATA
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
half = np.asarray(
    meta["target"]["half_extents_m"],
    dtype=float
)

with h5py.File(HDF5_PATH, "r") as f:
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

# ------------------------------------------------------------
# BACK-PROJECT TO WORLD COORDINATES
# ------------------------------------------------------------

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

observed = o3d.geometry.PointCloud()
observed.points = o3d.utility.Vector3dVector(points_world)

# ------------------------------------------------------------
# BUILD TEMPLATE
# ------------------------------------------------------------

mesh = o3d.geometry.TriangleMesh.create_box(
    width=2 * half[0],
    height=2 * half[1],
    depth=2 * half[2]
)

mesh.translate(-half)
mesh.compute_vertex_normals()

template = mesh.sample_points_uniformly(50000)

source = template.voxel_down_sample(VOXEL_SIZE)
target = observed.voxel_down_sample(VOXEL_SIZE)

source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015,
        max_nn=30
    )
)

target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015,
        max_nn=30
    )
)

# ------------------------------------------------------------
# PCA OF OBSERVED CLOUD
# ------------------------------------------------------------

P = np.asarray(target.points)

centroid = P.mean(axis=0)
centered = P - centroid

covariance = np.cov(centered.T)

eigenvalues, eigenvectors = np.linalg.eigh(covariance)

order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\n========== PCA DIAGNOSTICS ==========")
print("Eigenvalues:", eigenvalues)
print("Principal axis:", eigenvectors[:, 0])
print("Observed centroid:", centroid)

# ------------------------------------------------------------
# GENERATE VALID ORIENTATION CANDIDATES
# ------------------------------------------------------------

candidates = []

for minor_order in [(1, 2), (2, 1)]:

    for sign_y, sign_z in itertools.product(
        [-1, 1], repeat=2
    ):

        R = np.column_stack((
            eigenvectors[:, 0],
            sign_y * eigenvectors[:, minor_order[0]],
            sign_z * eigenvectors[:, minor_order[1]]
        ))

        # Ensure a proper rotation.
        if np.linalg.det(R) < 0:
            continue

        T_init = np.eye(4)
        T_init[:3, :3] = R

        # Align template origin with observed centroid.
        T_init[:3, 3] = centroid

        candidates.append(T_init)

print("\nValid orientation candidates:", len(candidates))

# ------------------------------------------------------------
# ICP FOR EACH CANDIDATE
# ------------------------------------------------------------

results = []

for i, T_init in enumerate(candidates):

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_DISTANCE,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    T_est = result.transformation

    translation_error = np.linalg.norm(
        T_est[:3, 3] - T_gt[:3, 3]
    )

    R_error = (
        T_est[:3, :3]
        @ T_gt[:3, :3].T
    )

    cos_angle = np.clip(
        (np.trace(R_error) - 1) / 2,
        -1,
        1
    )

    rotation_error = np.degrees(
        np.arccos(cos_angle)
    )

    results.append({
        "candidate": i,
        "fitness": result.fitness,
        "rmse_mm": result.inlier_rmse * 1000,
        "translation_error_mm": translation_error * 1000,
        "rotation_error_deg": rotation_error,
        "transformation": T_est
    })

    print(
        f"Candidate {i}: "
        f"fitness={result.fitness:.4f}, "
        f"RMSE={result.inlier_rmse * 1000:.2f} mm, "
        f"translation={translation_error * 1000:.2f} mm, "
        f"rotation={rotation_error:.2f} deg"
    )

# ------------------------------------------------------------
# RANK CANDIDATES
# ------------------------------------------------------------

results.sort(
    key=lambda x: (
        -x["fitness"],
        x["rmse_mm"]
    )
)

best = results[0]

print("\n========== BEST PCA + ICP RESULT ==========")
print("Candidate:", best["candidate"])
print("Fitness:", best["fitness"])
print("RMSE (mm):", best["rmse_mm"])
print("Translation error (mm):", best["translation_error_mm"])
print("Rotation error (deg):", best["rotation_error_deg"])
print("Transformation:\n", best["transformation"])

# ------------------------------------------------------------
# SAVE BEST ALIGNMENT
# ------------------------------------------------------------

aligned = o3d.geometry.PointCloud(template)
aligned.transform(best["transformation"])

o3d.io.write_point_cloud(
    str(OUTPUT_DIR / "observed.ply"),
    observed
)

o3d.io.write_point_cloud(
    str(OUTPUT_DIR / "estimated_template.ply"),
    aligned
)

# Save scalar results.
summary = {
    key: value
    for key, value in best.items()
    if key != "transformation"
}

summary["transformation"] = (
    best["transformation"].tolist()
)

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nResults saved to:", OUTPUT_DIR)