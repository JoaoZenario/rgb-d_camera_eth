
import json
import h5py
import numpy as np
import open3d as o3d

from pathlib import Path

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"
OUTPUT_DIR = Path("outputs/icp_convergence")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

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

mesh = o3d.geometry.TriangleMesh.create_box(
    width=2 * half[0],
    height=2 * half[1],
    depth=2 * half[2]
)
mesh.translate(-half)

template = mesh.sample_points_uniformly(50000)

# Use a common downsampled source and target for all trials.
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

def rotation_from_axis_angle(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    angle = np.deg2rad(angle_deg)

    x, y, z = axis
    Kx = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    return (
        np.eye(3)
        + np.sin(angle) * Kx
        + (1 - np.cos(angle)) * (Kx @ Kx)
    )

def pose_error(T_est, T_ref):
    translation = np.linalg.norm(
        T_est[:3, 3] - T_ref[:3, 3]
    )

    R = T_est[:3, :3] @ T_ref[:3, :3].T
    cos_angle = np.clip(
        (np.trace(R) - 1) / 2,
        -1,
        1
    )

    rotation = np.degrees(np.arccos(cos_angle))
    return translation, rotation

# Rotation axes are varied to avoid testing only one direction.
rotation_axes = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
]

translation_levels_mm = [5, 10, 20, 40]
rotation_levels_deg = [2, 5, 10, 15]

results = []

print("\nRunning ICP convergence basin experiment...")

for t_mm in translation_levels_mm:
    for r_deg in rotation_levels_deg:
        axis = np.asarray(
            rotation_axes[len(results) % len(rotation_axes)],
            dtype=float
        )

        # Deterministic random translation direction.
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        delta_R = rotation_from_axis_angle(axis, r_deg)

        T_perturbed = np.eye(4)
        T_perturbed[:3, :3] = delta_R @ T_gt[:3, :3]
        T_perturbed[:3, 3] = (
            T_gt[:3, 3] + direction * (t_mm / 1000.0)
        )

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            0.020,
            T_perturbed,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        trans_err, rot_err = pose_error(
            result.transformation,
            T_gt
        )

        success = (
            trans_err <= 0.005
            and rot_err <= 5.0
        )

        row = {
            "initial_translation_perturbation_mm": t_mm,
            "initial_rotation_perturbation_deg": r_deg,
            "final_translation_error_mm": trans_err * 1000,
            "final_rotation_error_deg": rot_err,
            "fitness": result.fitness,
            "inlier_rmse_mm": result.inlier_rmse * 1000,
            "success": success
        }

        results.append(row)

        print(
            f"Init {t_mm:2d} mm / {r_deg:2d} deg | "
            f"Final {trans_err*1000:7.2f} mm / "
            f"{rot_err:6.2f} deg | "
            f"Fitness {result.fitness:.3f} | "
            f"Success {success}"
        )

import csv

csv_path = OUTPUT_DIR / "convergence_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:", csv_path)
print("Successful trials:", sum(r["success"] for r in results))
print("Total trials:", len(results))
