import csv
import json
import h5py
import numpy as np

from pathlib import Path
from scripts.run_obb_percentile_sweep import estimate_obb
from src.sensor.depth_noise import apply_depth_noise

HDF5_PATH = "outputs/manufacturing_scene/0.hdf5"
META_PATH = "outputs/manufacturing_scene/ground_truth.json"

OUTPUT_DIR = Path("outputs/disparity_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = range(20)
SIGMAS = [0.03, 0.06, 0.12]

with h5py.File(HDF5_PATH, "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(f["category_id_segmaps"][...])

rows = []

for sigma in SIGMAS:
    for condition in ["disparity_only", "full_corruption"]:

        print(f"\n{condition.upper()} — sigma={sigma:.3f} px")

        for seed in SEEDS:

            noisy, _, _ = apply_depth_noise(
                depth,
                segmentation,
                seed=seed,
                disparity_sigma_px=sigma,
                gaussian_noise=True,
                range_missing=(condition == "full_corruption"),
                boundary_failures=(condition == "full_corruption")
            )

            result = estimate_obb(noisy, 0.0)

            invalid_rate = 100.0 * (
                1.0 - result["point_count"] /
                np.count_nonzero(segmentation == 4)
            )

            row = {
                "condition": condition,
                "disparity_sigma_px": sigma,
                "seed": seed,
                "invalid_target_percent": invalid_rate,
                **result
            }

            rows.append(row)

        subset = [
            r for r in rows
            if r["condition"] == condition
            and r["disparity_sigma_px"] == sigma
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
            f"Translation: {errors.mean():.3f} "
            f"+/- {errors.std(ddof=1):.3f} mm"
        )
        print(f"Long-axis error: {axes.mean():.3f} deg")
        print(f"Success: {100 * success:.1f}%")

with open(OUTPUT_DIR / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", OUTPUT_DIR / "results.csv")