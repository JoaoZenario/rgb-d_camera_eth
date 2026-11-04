import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import apply_depth_noise


HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/obb_percentile_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = range(20)
PERCENTILES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

with open(META_PATH) as f:
    meta = json.load(f)

K = np.asarray(meta["camera"]["intrinsics"], dtype=float)
T_cw = np.asarray(
    meta["camera"]["camera_to_world_blender"], dtype=float
)
T_gt = np.asarray(
    meta["target"]["object_to_world"], dtype=float
)

R_gt = T_gt[:3, :3]
t_gt = T_gt[:3, 3]

with h5py.File(HDF5_PATH, "r") as f:
    depth_ideal = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

target_mask = segmentation == 4
valid_ideal = (
    target_mask
    & np.isfinite(depth_ideal)
    & (depth_ideal > 0)
    & (depth_ideal < 1000)
)


def estimate_obb(depth, percentile):

    valid = (
        target_mask
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)
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

    mean = points_world.mean(axis=0)
    centered = points_world - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, order]

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    local = centered @ R

    lower = np.percentile(local, percentile, axis=0)
    upper = np.percentile(local, 100.0 - percentile, axis=0)

    local_mid = (lower + upper) / 2.0
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
        "axis_error_deg": float(axis_error),
        "success": bool(
            translation_error <= 0.005
            and axis_error <= 5.0
        )
    }


# Generate each noisy depth map once so every percentile
# is evaluated on exactly the same observations.
conditions = [("ideal", -1, depth_ideal)]

for seed in SEEDS:
    noisy, _, _ = apply_depth_noise(
        depth_ideal,
        segmentation,
        seed=seed
    )
    conditions.append(("full_corruption", seed, noisy))


rows = []

for condition, seed, depth in conditions:

    for percentile in PERCENTILES:

        result = estimate_obb(depth, percentile)

        rows.append({
            "condition": condition,
            "seed": seed,
            "percentile": percentile,
            **result
        })


print("\n========== PERCENTILE SWEEP ==========")

for condition in ["ideal", "full_corruption"]:

    print(f"\n{condition.upper()}")

    for percentile in PERCENTILES:

        subset = [
            r for r in rows
            if r["condition"] == condition
            and r["percentile"] == percentile
        ]

        errors = np.array([
            r["translation_error_mm"] for r in subset
        ])

        axes = np.array([
            r["axis_error_deg"] for r in subset
        ])

        success = np.mean([
            r["success"] for r in subset
        ])

        print(
            f"p={percentile:>4.2f}% | "
            f"translation={np.mean(errors):.3f} "
            f"+/- {np.std(errors, ddof=1) if len(errors) > 1 else 0:.3f} mm | "
            f"axis={np.mean(axes):.3f} deg | "
            f"success={100 * success:.1f}%"
        )


csv_path = OUTPUT_DIR / "results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "seed",
            "percentile",
            "point_count",
            "translation_error_mm",
            "axis_error_deg",
            "success"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", csv_path)