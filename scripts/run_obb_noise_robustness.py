import csv
import json
import h5py
import numpy as np

from pathlib import Path

from src.sensor.depth_noise import apply_depth_noise


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/obb_noise_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CATEGORY = 4
SEEDS = range(20)

POSITION_TOLERANCE_M = 0.005
AXIS_TOLERANCE_DEG = 5.0
MINIMUM_POINTS = 100


# ------------------------------------------------------------
# LOAD SCENE AND GROUND TRUTH
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

R_gt = T_gt[:3, :3]
t_gt = T_gt[:3, 3]

with h5py.File(HDF5_PATH, "r") as f:
    depth_ideal = np.squeeze(
        f["depth"][...]
    ).astype(np.float32)

    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

target_mask = segmentation == TARGET_CATEGORY

valid_ideal = (
    target_mask
    & np.isfinite(depth_ideal)
    & (depth_ideal > 0)
    & (depth_ideal < 1000)
)

ideal_target_count = int(valid_ideal.sum())


# ------------------------------------------------------------
# PCA-OBB ESTIMATOR
# ------------------------------------------------------------

def estimate_pca_obb(depth):

    valid = (
        target_mask
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)
    z = depth[v, u]

    if len(z) < MINIMUM_POINTS:
        raise ValueError(
            f"Insufficient valid target points: {len(z)}"
        )

    # Back-project to the OpenCV camera frame.
    points_cv = np.column_stack((
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z
    ))

    # Convert OpenCV camera coordinates to Blender camera coordinates.
    S = np.diag([1.0, -1.0, -1.0])
    points_camera_blender = (S @ points_cv.T).T

    # Transform into world coordinates.
    points_world = (
        T_cw[:3, :3] @ points_camera_blender.T
    ).T + T_cw[:3, 3]

    # PCA.
    sample_mean = points_world.mean(axis=0)
    centered = points_world - sample_mean

    covariance = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    R_pca = eigenvectors[:, order]

    # Ensure a right-handed coordinate frame.
    if np.linalg.det(R_pca) < 0:
        R_pca[:, 2] *= -1

    # Estimate centre from the midpoint of the projected bounds.
    local_points = centered @ R_pca

    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)

    local_midpoint = (local_min + local_max) / 2.0

    estimated_center = sample_mean + R_pca @ local_midpoint

    # Compare the undirected long axes.
    estimated_long_axis = R_pca[:, 0]
    true_long_axis = R_gt[:, 0]

    axis_cosine = np.clip(
        abs(np.dot(estimated_long_axis, true_long_axis)),
        0.0,
        1.0
    )

    axis_error_deg = np.degrees(
        np.arccos(axis_cosine)
    )

    translation_error_m = np.linalg.norm(
        estimated_center - t_gt
    )

    estimated_extents = local_max - local_min

    return {
        "point_count": int(len(points_world)),
        "translation_error_mm": float(
            translation_error_m * 1000
        ),
        "long_axis_error_deg": float(axis_error_deg),
        "grasp_axis_success": bool(
            translation_error_m <= POSITION_TOLERANCE_M
            and axis_error_deg <= AXIS_TOLERANCE_DEG
        ),
        "estimated_extents_m": estimated_extents.tolist(),
        "eigenvalues": eigenvalues.tolist()
    }


# ------------------------------------------------------------
# RUN IDEAL BASELINE
# ------------------------------------------------------------

print("\n========== IDEAL DEPTH BASELINE ==========")

ideal_result = estimate_pca_obb(depth_ideal)

ideal_result["condition"] = "ideal"
ideal_result["seed"] = -1

print("Valid target points:", ideal_result["point_count"])
print(
    "Translation error:",
    f'{ideal_result["translation_error_mm"]:.3f} mm'
)
print(
    "Long-axis error:",
    f'{ideal_result["long_axis_error_deg"]:.3f} deg'
)
print("Grasp-axis success:", ideal_result["grasp_axis_success"])


# ------------------------------------------------------------
# RUN CORRUPTED DEPTH EXPERIMENT
# ------------------------------------------------------------

results = [ideal_result]

print("\n========== CORRUPTED DEPTH EXPERIMENT ==========")

for seed in SEEDS:

    noisy_depth, invalid_mask, edge_mask = apply_depth_noise(
        depth_ideal,
        segmentation,
        seed=seed
    )

    try:
        result = estimate_pca_obb(noisy_depth)

        result["condition"] = "full_corruption"
        result["seed"] = int(seed)

        result["target_invalid_rate_pct"] = float(
            100.0 * (
                ideal_target_count - result["point_count"]
            ) / ideal_target_count
        )

        results.append(result)

        print(
            f"Seed {seed:02d}: "
            f"points={result['point_count']:4d}, "
            f"invalid={result['target_invalid_rate_pct']:.2f}%, "
            f"translation={result['translation_error_mm']:.3f} mm, "
            f"axis={result['long_axis_error_deg']:.3f} deg, "
            f"success={result['grasp_axis_success']}"
        )

    except Exception as exc:
        print(f"Seed {seed:02d}: FAILED — {exc}")

        results.append({
            "condition": "full_corruption",
            "seed": int(seed),
            "point_count": 0,
            "translation_error_mm": None,
            "long_axis_error_deg": None,
            "grasp_axis_success": False,
            "target_invalid_rate_pct": None,
            "failure": str(exc)
        })


# ------------------------------------------------------------
# SUMMARISE CORRUPTED RESULTS
# ------------------------------------------------------------

successful_runs = [
    r for r in results
    if r["condition"] == "full_corruption"
    and r["translation_error_mm"] is not None
]

print("\n========== SUMMARY ==========")

if successful_runs:

    translation_errors = np.array([
        r["translation_error_mm"]
        for r in successful_runs
    ])

    axis_errors = np.array([
        r["long_axis_error_deg"]
        for r in successful_runs
    ])

    success_count = sum(
        r["grasp_axis_success"]
        for r in successful_runs
    )

    print("Completed runs:", len(successful_runs), "/", len(SEEDS))

    print(
        "Translation error (mm): "
        f"mean={np.mean(translation_errors):.3f}, "
        f"std={np.std(translation_errors, ddof=1):.3f}, "
        f"median={np.median(translation_errors):.3f}, "
        f"P95={np.percentile(translation_errors, 95):.3f}"
    )

    print(
        "Long-axis error (deg): "
        f"mean={np.mean(axis_errors):.3f}, "
        f"std={np.std(axis_errors, ddof=1):.3f}, "
        f"median={np.median(axis_errors):.3f}, "
        f"P95={np.percentile(axis_errors, 95):.3f}"
    )

    print(
        "Grasp-axis success rate:",
        f"{100 * success_count / len(successful_runs):.1f}%"
    )

    summary = {
        "ideal_baseline": ideal_result,
        "corrupted_runs": len(successful_runs),
        "requested_runs": len(SEEDS),
        "translation_error_mm": {
            "mean": float(np.mean(translation_errors)),
            "std": float(np.std(translation_errors, ddof=1)),
            "median": float(np.median(translation_errors)),
            "p95": float(np.percentile(translation_errors, 95))
        },
        "long_axis_error_deg": {
            "mean": float(np.mean(axis_errors)),
            "std": float(np.std(axis_errors, ddof=1)),
            "median": float(np.median(axis_errors)),
            "p95": float(np.percentile(axis_errors, 95))
        },
        "grasp_axis_success_rate": float(
            success_count / len(successful_runs)
        ),
        "position_tolerance_mm": POSITION_TOLERANCE_M * 1000,
        "axis_tolerance_deg": AXIS_TOLERANCE_DEG,
        "physical_calibration_performed": False
    }

else:
    summary = {
        "error": "No corrupted runs completed successfully."
    }


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "results.csv"

fieldnames = [
    "condition",
    "seed",
    "point_count",
    "target_invalid_rate_pct",
    "translation_error_mm",
    "long_axis_error_deg",
    "grasp_axis_success",
    "failure"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(results)

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved:", csv_path)
print("Saved:", OUTPUT_DIR / "summary.json")