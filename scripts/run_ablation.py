import os
import csv

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.sensor.depth_noise import apply_depth_noise

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
CSV_FILE = "report/ablation_results.csv"
FIGURE_FILE = "report/figures/ablation_comparison_001.png"

SEED = 42

CONFIGURATIONS = {
    "Baseline": {
        "gaussian_noise": False,
        "range_missing": False,
        "boundary_failures": False,
    },
    "A: Gaussian": {
        "gaussian_noise": True,
        "range_missing": False,
        "boundary_failures": False,
    },
    "B: Range missing": {
        "gaussian_noise": False,
        "range_missing": True,
        "boundary_failures": False,
    },
    "C: Boundary failures": {
        "gaussian_noise": False,
        "range_missing": False,
        "boundary_failures": True,
    },
    "D: Full model": {
        "gaussian_noise": True,
        "range_missing": True,
        "boundary_failures": True,
    },
}


# --------------------------------------------------
# Load ideal depth and segmentation
# --------------------------------------------------

with h5py.File(INPUT_FILE, "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

if depth.shape != segmentation.shape:
    raise ValueError("Depth and segmentation shapes do not match.")

valid_original = (
    np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

original_valid_count = np.count_nonzero(valid_original)

print("Depth shape:", depth.shape)
print("Segmentation shape:", segmentation.shape)
print("Original valid pixels:", original_valid_count)


# --------------------------------------------------
# Run ablation configurations
# --------------------------------------------------

results = []
corrupted_maps = {}

for name, settings in CONFIGURATIONS.items():

    corrupted, invalid_mask, edge_mask = apply_depth_noise(
        depth,
        segmentation,
        seed=SEED,
        **settings,
    )

    corrupted_maps[name] = corrupted.copy()

    valid_corrupted = (
        np.isfinite(corrupted)
        & (corrupted > 0)
        & (corrupted < 1000)
    )

    common_valid = valid_original & valid_corrupted

    remaining_count = np.count_nonzero(valid_corrupted)
    newly_invalid_count = np.count_nonzero(
        valid_original & ~valid_corrupted
    )

    invalid_rate = (
        newly_invalid_count / original_valid_count * 100
    )

    # Error metrics use only pixels valid in both maps.
    errors = corrupted[common_valid] - depth[common_valid]

    if errors.size > 0:
        mae = np.mean(np.abs(errors)) * 1000
        rmse = np.sqrt(np.mean(errors**2)) * 1000
        bias = np.mean(errors) * 1000
    else:
        mae = np.nan
        rmse = np.nan
        bias = np.nan

    result = {
        "Configuration": name,
        "Remaining valid pixels": remaining_count,
        "Newly invalid pixels": newly_invalid_count,
        "Invalid rate (%)": invalid_rate,
        "MAE (mm)": mae,
        "RMSE (mm)": rmse,
        "Bias (mm)": bias,
    }

    results.append(result)


# --------------------------------------------------
# Paired comparison: Gaussian-only vs full model
# --------------------------------------------------

gaussian_depth = corrupted_maps["A: Gaussian"]
full_depth = corrupted_maps["D: Full model"]

# Only pixels with valid measurements in BOTH maps.
paired_valid = (
    valid_original
    & np.isfinite(gaussian_depth)
    & (gaussian_depth > 0)
    & (gaussian_depth < 1000)
    & np.isfinite(full_depth)
    & (full_depth > 0)
    & (full_depth < 1000)
)

gaussian_errors = (
    gaussian_depth[paired_valid] - depth[paired_valid]
)

full_errors = (
    full_depth[paired_valid] - depth[paired_valid]
)

def calculate_metrics(errors):
    return {
        "MAE (mm)": np.mean(np.abs(errors)) * 1000,
        "RMSE (mm)": np.sqrt(np.mean(errors**2)) * 1000,
        "Bias (mm)": np.mean(errors) * 1000,
    }

paired_gaussian = calculate_metrics(gaussian_errors)
paired_full = calculate_metrics(full_errors)

print("\n========== PAIRED COMPARISON ==========")
print("Common valid pixels:", np.count_nonzero(paired_valid))

print("\nGaussian-only:")
for metric, value in paired_gaussian.items():
    print(f"{metric}: {value:.2f}")

print("\nFull model:")
for metric, value in paired_full.items():
    print(f"{metric}: {value:.2f}")


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n========== ABLATION RESULTS ==========\n")

print(
    f"{'Configuration':<23}"
    f"{'Invalid (%)':>13}"
    f"{'MAE (mm)':>12}"
    f"{'RMSE (mm)':>12}"
    f"{'Bias (mm)':>12}"
)

for r in results:
    print(
        f"{r['Configuration']:<23}"
        f"{r['Invalid rate (%)']:>13.2f}"
        f"{r['MAE (mm)']:>12.2f}"
        f"{r['RMSE (mm)']:>12.2f}"
        f"{r['Bias (mm)']:>12.2f}"
    )


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

os.makedirs("report", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nCSV saved to: {CSV_FILE}")


# --------------------------------------------------
# Plot invalid rate and RMSE
# --------------------------------------------------

names = [r["Configuration"] for r in results]
invalid_rates = [r["Invalid rate (%)"] for r in results]
rmses = [r["RMSE (mm)"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].bar(names, invalid_rates)
axes[0].set_title("Invalid depth rate")
axes[0].set_ylabel("Invalid pixels (%)")
axes[0].tick_params(axis="x", rotation=25)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(names, rmses)
axes[1].set_title("Depth RMSE")
axes[1].set_ylabel("RMSE (mm)")
axes[1].tick_params(axis="x", rotation=25)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
plt.close()

print(f"Figure saved to: {FIGURE_FILE}")