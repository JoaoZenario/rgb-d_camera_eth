from pathlib import Path
import csv

import h5py
import numpy as np

from src.sensor.depth_noise import apply_depth_noise


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = Path("outputs/manufacturing_scene/0.hdf5")
OUTPUT_DIR = Path("outputs/multiseed_centroid_experiment")

# Match the intrinsics used by the current perception experiment.
# These are the current Blender-rendering intrinsics, not measured
# physical D455 calibration parameters.
FX = 888.88909234
FY = 888.88909234
CX = 319.5
CY = 239.5

PROFILE_CATEGORY_ID = 2
SEEDS = range(10, 40)  # 30 independent trials


# ------------------------------------------------------------
# Point-cloud utilities
# ------------------------------------------------------------

def backproject(depth, mask):
    """Back-project selected valid depth pixels into camera coordinates."""
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)
    z = depth[v, u].astype(np.float64)

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    return np.column_stack((x, y, z)), valid


def centroid(points):
    if len(points) == 0:
        return np.full(3, np.nan)
    return points.mean(axis=0)


def vector_norm_mm(a, b):
    return float(np.linalg.norm(a - b) * 1000.0)


# ------------------------------------------------------------
# Load scene
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

with h5py.File(INPUT_FILE, "r") as f:
    depth_ideal = f["depth"][:].astype(np.float32)
    segmentation = f["category_id_segmaps"][:]

profile_mask = segmentation == PROFILE_CATEGORY_ID

if not np.any(profile_mask):
    raise RuntimeError(
        f"No pixels found for category ID {PROFILE_CATEGORY_ID}. "
        "Check the segmentation IDs in the HDF5 file."
    )

ideal_points, ideal_valid = backproject(depth_ideal, profile_mask)

if len(ideal_points) == 0:
    raise RuntimeError("The selected profile has no valid ideal depth pixels.")

ideal_centroid = centroid(ideal_points)

print(f"Input: {INPUT_FILE}")
print(f"Profile pixels: {int(profile_mask.sum())}")
print(f"Valid ideal points: {len(ideal_points)}")
print(f"Ideal centroid: {ideal_centroid}")
print(f"Seeds: {len(SEEDS)}")


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

conditions = {
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

rows = []

for seed in SEEDS:
    print(f"\nSeed {seed}")

    for condition, options in conditions.items():

        if condition == "ideal":
            corrupted = depth_ideal.copy()
            invalid_mask = np.zeros(depth_ideal.shape, dtype=bool)

        else:
            corrupted, invalid_mask, _ = apply_depth_noise(
                depth_ideal,
                segmentation,
                seed=seed,
                **options,
            )

        points, condition_valid = backproject(corrupted, profile_mask)

        if len(points) == 0:
            print(f"  WARNING: {condition} produced no valid points.")
            continue

        condition_centroid = centroid(points)

        # Metric A: total observed centroid shift relative to the
        # centroid of the complete ideal point population.
        population_shift_mm = vector_norm_mm(
            condition_centroid,
            ideal_centroid,
        )

        # Metric B: compare corrupted and ideal depths on exactly
        # the same surviving pixels. This isolates depth changes
        # from changes in which pixels contribute.
        common_valid = ideal_valid & condition_valid

        ideal_common_points, _ = backproject(
            depth_ideal,
            profile_mask & common_valid,
        )
        corrupted_common_points, _ = backproject(
            corrupted,
            profile_mask & common_valid,
        )

        ideal_common_centroid = centroid(ideal_common_points)
        corrupted_common_centroid = centroid(corrupted_common_points)

        paired_depth_shift_mm = vector_norm_mm(
            corrupted_common_centroid,
            ideal_common_centroid,
        )

        # Count invalid profile pixels, not invalid background pixels.
        profile_pixel_count = int(profile_mask.sum())
        surviving_profile_pixels = len(points)
        missing_fraction = (
            1.0 - surviving_profile_pixels / profile_pixel_count
        )

        rows.append({
            "seed": seed,
            "condition": condition,
            "valid_points": surviving_profile_pixels,
            "missing_fraction": missing_fraction,
            "population_centroid_shift_mm": population_shift_mm,
            "paired_depth_centroid_shift_mm": paired_depth_shift_mm,
            "centroid_x_mm": condition_centroid[0] * 1000,
            "centroid_y_mm": condition_centroid[1] * 1000,
            "centroid_z_mm": condition_centroid[2] * 1000,
        })

        print(
            f"  {condition:12s} "
            f"valid={surviving_profile_pixels:6d} "
            f"missing={100 * missing_fraction:5.2f}% "
            f"population_shift={population_shift_mm:7.3f} mm "
            f"paired_depth_shift={paired_depth_shift_mm:7.3f} mm"
        )


# ------------------------------------------------------------
# Save individual trials
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_file = OUTPUT_DIR / "per_seed_results.csv"

fieldnames = list(rows[0].keys())

with raw_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------
# Summarise across seeds
# ------------------------------------------------------------

summary_file = OUTPUT_DIR / "summary.csv"

summary_fields = [
    "condition",
    "trials",
    "mean_missing_percent",
    "mean_population_shift_mm",
    "std_population_shift_mm",
    "p95_population_shift_mm",
    "mean_paired_depth_shift_mm",
    "std_paired_depth_shift_mm",
    "p95_paired_depth_shift_mm",
]

summary_rows = []

for condition in conditions:
    selected = [r for r in rows if r["condition"] == condition]

    if not selected:
        continue

    population = np.array([
        r["population_centroid_shift_mm"] for r in selected
    ])
    paired = np.array([
        r["paired_depth_centroid_shift_mm"] for r in selected
    ])
    missing = np.array([
        r["missing_fraction"] for r in selected
    ])

    summary_rows.append({
        "condition": condition,
        "trials": len(selected),
        "mean_missing_percent": 100 * missing.mean(),
        "mean_population_shift_mm": population.mean(),
        "std_population_shift_mm": population.std(ddof=1),
        "p95_population_shift_mm": np.percentile(population, 95),
        "mean_paired_depth_shift_mm": paired.mean(),
        "std_paired_depth_shift_mm": paired.std(ddof=1),
        "p95_paired_depth_shift_mm": np.percentile(paired, 95),
    })

with summary_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MULTI-SEED CENTROID EXPERIMENT SUMMARY")
print("=" * 90)

for result in summary_rows:
    print(f"\n{result['condition'].upper()}")
    print(
        f"  Missing pixels:          "
        f"{result['mean_missing_percent']:.2f}%"
    )
    print(
        f"  Population shift:        "
        f"{result['mean_population_shift_mm']:.3f} "
        f"+/- {result['std_population_shift_mm']:.3f} mm"
    )
    print(
        f"  Paired depth shift:      "
        f"{result['mean_paired_depth_shift_mm']:.3f} "
        f"+/- {result['std_paired_depth_shift_mm']:.3f} mm"
    )
    print(
        f"  Population shift P95:    "
        f"{result['p95_population_shift_mm']:.3f} mm"
    )
    print(
        f"  Paired depth shift P95:  "
        f"{result['p95_paired_depth_shift_mm']:.3f} mm"
    )

print(f"\nPer-seed results: {raw_file}")
print(f"Summary:          {summary_file}")