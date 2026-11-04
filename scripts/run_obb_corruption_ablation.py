import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import apply_depth_noise


HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/obb_corruption_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = range(20)
TARGET_CATEGORY = 4

POSITION_TOLERANCE_M = 0.005
AXIS_TOLERANCE_DEG = 5.0

CONDITIONS = {
    "ideal": (False, False, False),
    "disparity_only": (True, False, False),
    "range_missing_only": (False, True, False),
    "boundary_failures_only": (False, False, True),
    "full_corruption": (True, True, True),
}


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

R_gt = T_gt[:3, :3]
t_gt = T_gt[:3, 3]

with h5py.File(HDF5_PATH, "r") as f:
    depth_ideal = np.squeeze(f["depth"][...]).astype(np.float32)
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

ideal_count = int(valid_ideal.sum())


# ------------------------------------------------------------
# PCA-OBB ESTIMATOR
# ------------------------------------------------------------

def estimate_obb(depth):

    valid = (
        target_mask
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)
    z = depth[v, u]

    if len(z) < 100:
        raise ValueError("Too few valid target points")

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

    mean = points_world.mean(axis=0)
    centered = points_world - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    R = eigenvectors[:, order]

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    local = centered @ R

    local_min = local.min(axis=0)
    local_max = local.max(axis=0)

    local_mid = (local_min + local_max) / 2.0
    centre = mean + R @ local_mid

    axis_cosine = np.clip(
        abs(np.dot(R[:, 0], R_gt[:, 0])),
        0.0,
        1.0
    )

    axis_error = np.degrees(np.arccos(axis_cosine))
    translation_error = np.linalg.norm(centre - t_gt)

    return {
        "point_count": len(z),
        "translation_error_mm": float(translation_error * 1000),
        "long_axis_error_deg": float(axis_error),
        "success": bool(
            translation_error <= POSITION_TOLERANCE_M
            and axis_error <= AXIS_TOLERANCE_DEG
        ),
        "extents_m": (local_max - local_min).tolist(),
        "eigenvalues": eigenvalues.tolist()
    }


# ------------------------------------------------------------
# EXPERIMENT
# ------------------------------------------------------------

rows = []

for condition, switches in CONDITIONS.items():

    gaussian, range_missing, boundary = switches

    seeds = [-1] if condition == "ideal" else list(SEEDS)

    print(f"\n========== {condition.upper()} ==========")

    for seed in seeds:

        if condition == "ideal":
            depth = depth_ideal

        else:
            depth, _, _ = apply_depth_noise(
                depth_ideal,
                segmentation,
                seed=seed,
                gaussian_noise=gaussian,
                range_missing=range_missing,
                boundary_failures=boundary
            )

        result = estimate_obb(depth)

        invalid_rate = (
            100.0 * (ideal_count - result["point_count"]) / ideal_count
        )

        row = {
            "condition": condition,
            "seed": seed,
            "point_count": result["point_count"],
            "invalid_rate_pct": invalid_rate,
            "translation_error_mm": result["translation_error_mm"],
            "long_axis_error_deg": result["long_axis_error_deg"],
            "success": result["success"],
            "extents_m": result["extents_m"],
            "eigenvalues": result["eigenvalues"]
        }

        rows.append(row)

        if seed in (-1, 0, 1, 2):
            print(
                f"seed={seed:>2}: "
                f"invalid={invalid_rate:.2f}%, "
                f"translation={result['translation_error_mm']:.3f} mm, "
                f"axis={result['long_axis_error_deg']:.3f} deg, "
                f"success={result['success']}"
            )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n========== CONDITION SUMMARY ==========")

summary = {}

for condition in CONDITIONS:

    subset = [
        r for r in rows
        if r["condition"] == condition
    ]

    translation = np.array([
        r["translation_error_mm"] for r in subset
    ])

    axis = np.array([
        r["long_axis_error_deg"] for r in subset
    ])

    success_rate = np.mean([
        r["success"] for r in subset
    ])

    invalid = np.array([
        r["invalid_rate_pct"] for r in subset
    ])

    stats = {
        "runs": len(subset),
        "invalid_rate_mean_pct": float(np.mean(invalid)),
        "translation_mean_mm": float(np.mean(translation)),
        "translation_std_mm": float(
            np.std(translation, ddof=1)
        ) if len(translation) > 1 else 0.0,
        "translation_median_mm": float(np.median(translation)),
        "translation_p95_mm": float(np.percentile(translation, 95)),
        "axis_mean_deg": float(np.mean(axis)),
        "axis_std_deg": float(
            np.std(axis, ddof=1)
        ) if len(axis) > 1 else 0.0,
        "success_rate": float(success_rate)
    }

    summary[condition] = stats

    print(
        f"{condition:24s} "
        f"translation={stats['translation_mean_mm']:.3f} "
        f"+/- {stats['translation_std_mm']:.3f} mm, "
        f"axis={stats['axis_mean_deg']:.3f} deg, "
        f"invalid={stats['invalid_rate_mean_pct']:.2f}%, "
        f"success={100 * success_rate:.1f}%"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

csv_path = OUTPUT_DIR / "results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "seed",
            "point_count",
            "invalid_rate_pct",
            "translation_error_mm",
            "long_axis_error_deg",
            "success",
            "extents_m",
            "eigenvalues"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nSaved:", csv_path)
print("Saved:", OUTPUT_DIR / "summary.json")