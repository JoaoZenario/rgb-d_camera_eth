import csv
import json
import h5py
import numpy as np
import open3d as o3d

from pathlib import Path
from scipy.spatial import cKDTree
from src.sensor.depth_noise import apply_depth_noise

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
METADATA_FILE = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/paired_pose_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = range(10, 20)

VOXEL_SIZE = 0.003
NORMAL_RADIUS = 0.015
ICP_DISTANCE = 0.020

# Controlled initial-pose uncertainty.
INITIAL_TRANSLATION_MM = 10.0
INITIAL_ROTATION_DEG = 5.0

# Simulated grasp-feasibility thresholds.
POSITION_TOLERANCE_MM = 5.0
ORIENTATION_TOLERANCE_DEG = 5.0

CONDITIONS = {
    "ideal": {
        "gaussian_noise": False,
        "range_missing": False,
        "boundary_failures": False,
    },
    "noise_only": {
        "gaussian_noise": True,
        "range_missing": False,
        "boundary_failures": False,
    },
    "dropout_only": {
        "gaussian_noise": False,
        "range_missing": True,
        "boundary_failures": True,
    },
    "full": {
        "gaussian_noise": True,
        "range_missing": True,
        "boundary_failures": True,
    },
}


# ============================================================
# LOAD SCENE AND GROUND TRUTH
# ============================================================

with open(METADATA_FILE) as f:
    metadata = json.load(f)

K = np.asarray(metadata["camera"]["intrinsics"], dtype=float)
T_camera_world = np.asarray(
    metadata["camera"]["camera_to_world_blender"],
    dtype=float
)
T_ground_truth = np.asarray(
    metadata["target"]["object_to_world"],
    dtype=float
)
half_extents = np.asarray(
    metadata["target"]["half_extents_m"],
    dtype=float
)

with h5py.File(INPUT_FILE, "r") as f:
    ideal_depth = np.asarray(f["depth"], dtype=np.float32)
    segmentation = np.asarray(f["category_id_segmaps"])

if ideal_depth.shape != segmentation.shape:
    raise ValueError("Depth and segmentation dimensions differ.")

target_mask = segmentation == 4

print("Image shape:", ideal_depth.shape)
print("Target pixels:", int(target_mask.sum()))


# ============================================================
# TEMPLATE MODEL IN OBJECT-LOCAL COORDINATES
# ============================================================

o3d.utility.random.seed(42)

mesh = o3d.geometry.TriangleMesh.create_box(
    width=2.0 * half_extents[0],
    height=2.0 * half_extents[1],
    depth=2.0 * half_extents[2]
)

mesh.translate(-half_extents)
mesh.compute_vertex_normals()

template = mesh.sample_points_uniformly(
    number_of_points=30000
)

source = template.voxel_down_sample(VOXEL_SIZE)

source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS,
        max_nn=30
    )
)

# Reuse the sampled template points for ADD-S evaluation.
model_points = np.asarray(template.points).copy()

print("Template points:", len(source.points))


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def backproject_to_world(depth_map):
    """Back-project valid target pixels into Blender world coordinates."""

    valid = (
        target_mask
        & np.isfinite(depth_map)
        & (depth_map > 0)
        & (depth_map < 1000)
    )

    v, u = np.where(valid)

    if len(u) < 100:
        return None, int(len(u))

    z = depth_map[v, u]

    points_cv = np.column_stack((
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z
    ))

    # OpenCV camera axes -> Blender camera axes.
    S = np.diag([1.0, -1.0, -1.0])
    points_camera_blender = (S @ points_cv.T).T

    points_world = (
        T_camera_world[:3, :3]
        @ points_camera_blender.T
    ).T + T_camera_world[:3, 3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_world)

    return cloud, int(len(u))


def rotation_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis

    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

    return (
        np.eye(3)
        + np.sin(angle_rad) * skew
        + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )


def make_initial_pose(seed):
    """Create one deterministic initial pose, shared by all conditions."""

    rng = np.random.default_rng(seed)

    translation_direction = rng.normal(size=3)
    translation_direction /= np.linalg.norm(
        translation_direction
    )

    rotation_axis = rng.normal(size=3)
    rotation_axis /= np.linalg.norm(rotation_axis)

    angle = np.deg2rad(INITIAL_ROTATION_DEG)
    delta_R = rotation_from_axis_angle(rotation_axis, angle)

    T_initial = np.eye(4)

    # Perturb in world coordinates.
    T_initial[:3, :3] = (
        delta_R @ T_ground_truth[:3, :3]
    )

    T_initial[:3, 3] = (
        T_ground_truth[:3, 3]
        + translation_direction
        * INITIAL_TRANSLATION_MM / 1000.0
    )

    return T_initial


def calculate_pose_errors(T_est):
    translation_error = np.linalg.norm(
        T_est[:3, 3] - T_ground_truth[:3, 3]
    )

    R_error = (
        T_est[:3, :3]
        @ T_ground_truth[:3, :3].T
    )

    cosine = np.clip(
        (np.trace(R_error) - 1.0) / 2.0,
        -1.0,
        1.0
    )

    rotation_error = np.degrees(np.arccos(cosine))

    # Symmetry-aware ADD-S.
    estimated_model = (
        T_est[:3, :3] @ model_points.T
    ).T + T_est[:3, 3]

    ground_truth_model = (
        T_ground_truth[:3, :3] @ model_points.T
    ).T + T_ground_truth[:3, 3]

    tree = cKDTree(ground_truth_model)
    distances, _ = tree.query(estimated_model, k=1)

    adds = float(np.mean(distances))

    return translation_error, rotation_error, adds


# ============================================================
# RUN PAIRED CONDITIONS
# ============================================================

results = []

for seed in SEEDS:

    print(f"\n========== SEED {seed} ==========")

    # Same initial pose for every condition in this seed.
    initial_pose = make_initial_pose(seed)

    for condition, settings in CONDITIONS.items():

        corrupted_depth, invalid_mask, edge_mask = (
            apply_depth_noise(
                ideal_depth,
                segmentation,
                seed=seed,
                **settings
            )
        )

        observed, valid_target_pixels = backproject_to_world(
            corrupted_depth
        )

        if observed is None:
            print(condition, ": insufficient target points")
            results.append({
                "seed": seed,
                "condition": condition,
                "valid_target_pixels": valid_target_pixels,
                "fitness": 0.0,
                "inlier_rmse_mm": float("nan"),
                "translation_error_mm": float("nan"),
                "rotation_error_deg": float("nan"),
                "adds_mm": float("nan"),
                "grasp_feasible": False,
            })
            continue

        target = observed.voxel_down_sample(VOXEL_SIZE)

        if len(target.points) < 30:
            print(condition, ": too few downsampled points")
            continue

        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS,
                max_nn=30
            )
        )

        registration = (
            o3d.pipelines.registration.registration_icp(
                source,
                target,
                ICP_DISTANCE,
                initial_pose,
                o3d.pipelines.registration
                .TransformationEstimationPointToPlane()
            )
        )

        T_est = registration.transformation

        trans_error, rot_error, adds = calculate_pose_errors(
            T_est
        )

        success = (
            trans_error * 1000 <= POSITION_TOLERANCE_MM
            and rot_error <= ORIENTATION_TOLERANCE_DEG
        )

        row = {
            "seed": seed,
            "condition": condition,
            "valid_target_pixels": valid_target_pixels,
            "fitness": float(registration.fitness),
            "inlier_rmse_mm": float(
                registration.inlier_rmse * 1000
            ),
            "translation_error_mm": float(
                trans_error * 1000
            ),
            "rotation_error_deg": float(rot_error),
            "adds_mm": float(adds * 1000),
            "grasp_feasible": bool(success),
        }

        results.append(row)

        print(
            f"{condition:12s} | "
            f"pixels {valid_target_pixels:4d} | "
            f"fitness {registration.fitness:.3f} | "
            f"translation {trans_error*1000:7.2f} mm | "
            f"rotation {rot_error:6.2f} deg | "
            f"ADD-S {adds*1000:7.2f} mm | "
            f"success {success}"
        )


# ============================================================
# SAVE PER-TRIAL RESULTS
# ============================================================

csv_path = OUTPUT_DIR / "per_trial_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARISE RESULTS
# ============================================================

summary = []

for condition in CONDITIONS:

    subset = [
        r for r in results
        if r["condition"] == condition
    ]

    def values(key):
        return np.asarray([
            r[key] for r in subset
            if np.isfinite(r[key])
        ], dtype=float)

    trans = values("translation_error_mm")
    rot = values("rotation_error_deg")
    adds = values("adds_mm")
    fitness = values("fitness")

    successes = np.asarray([
        r["grasp_feasible"] for r in subset
    ], dtype=bool)

    row = {
        "condition": condition,
        "trials": len(subset),
        "successful_trials": int(successes.sum()),
        "success_rate_percent": (
            float(successes.mean() * 100)
            if len(successes) else float("nan")
        ),
        "translation_mean_mm": (
            float(trans.mean()) if len(trans) else float("nan")
        ),
        "translation_median_mm": (
            float(np.median(trans)) if len(trans) else float("nan")
        ),
        "rotation_mean_deg": (
            float(rot.mean()) if len(rot) else float("nan")
        ),
        "adds_mean_mm": (
            float(adds.mean()) if len(adds) else float("nan")
        ),
        "fitness_mean": (
            float(fitness.mean()) if len(fitness) else float("nan")
        )
    }

    summary.append(row)


summary_path = OUTPUT_DIR / "summary.json"

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========== PAIRED POSE EXPERIMENT SUMMARY ==========")

for row in summary:
    print(f"\n{row['condition']}")
    print("Trials:", row["trials"])
    print("Success rate (%):", row["success_rate_percent"])
    print("Mean translation error (mm):",
          row["translation_mean_mm"])
    print("Median translation error (mm):",
          row["translation_median_mm"])
    print("Mean rotation error (deg):",
          row["rotation_mean_deg"])
    print("Mean ADD-S (mm):", row["adds_mean_mm"])
    print("Mean fitness:", row["fitness_mean"])

print("\nPer-trial results:", csv_path)
print("Summary:", summary_path)