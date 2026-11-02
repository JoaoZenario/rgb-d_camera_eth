
import os
import glob
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.sensor.depth_noise import apply_depth_noise


# --------------------------------------------------
# 1. Load ideal BlenderProc output
# --------------------------------------------------

with h5py.File(
    "outputs/manufacturing_scene/0.hdf5", "r"
) as f:
    depth = np.squeeze(f["depth"][...])
    segmentation = np.squeeze(f["category_id_segmaps"][...])

print("Depth shape:", depth.shape)
print("Segmentation shape:", segmentation.shape)


# --------------------------------------------------
# 2. Apply simulated sensor imperfections
# --------------------------------------------------

noisy_depth, invalid_mask, edge_mask = apply_depth_noise(
    depth,
    segmentation,
    seed=42
)


# --------------------------------------------------
# 3. Define valid measurements
# --------------------------------------------------

valid_original = (
    np.isfinite(depth) &
    (depth > 0) &
    (depth < 1000)
)

valid_noisy = (
    np.isfinite(noisy_depth) &
    (noisy_depth > 0) &
    (noisy_depth < 1000)
)

common_valid = valid_original & valid_noisy
newly_invalid = valid_original & ~valid_noisy


# --------------------------------------------------
# 4. Overall quantitative evaluation
# --------------------------------------------------

errors = noisy_depth[common_valid] - depth[common_valid]

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
bias = np.mean(errors)

invalid_percentage = (
    100 * np.sum(newly_invalid) / np.sum(valid_original)
)

print("\n========== OVERALL EVALUATION ==========")
print(f"Original valid pixels: {np.sum(valid_original)}")
print(f"Remaining valid pixels: {np.sum(common_valid)}")
print(f"Newly invalid pixels: {np.sum(newly_invalid)}")
print(f"Invalid depth rate: {invalid_percentage:.2f}%")
print(f"MAE: {mae * 1000:.2f} mm")
print(f"RMSE: {rmse * 1000:.2f} mm")
print(f"Mean signed error (bias): {bias * 1000:.2f} mm")


# --------------------------------------------------
# 5. Range-dependent evaluation
# --------------------------------------------------

range_bins = [
    (0, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6)
]

print("\n========== RANGE-DEPENDENT EVALUATION ==========")
print(
    f"{'Range (m)':<12}"
    f"{'Pixels':<10}"
    f"{'Invalid (%)':<14}"
    f"{'MAE (mm)':<12}"
    f"{'RMSE (mm)':<12}"
)

for lower, upper in range_bins:

    range_mask = (
        valid_original &
        (depth >= lower) &
        (depth < upper)
    )

    total_pixels = np.sum(range_mask)

    if total_pixels == 0:
        continue

    range_invalid = range_mask & ~valid_noisy

    range_invalid_rate = (
        100 * np.sum(range_invalid) / total_pixels
    )

    range_common = range_mask & valid_noisy

    range_errors = (
        noisy_depth[range_common] - depth[range_common]
    )

    if len(range_errors) > 0:
        range_mae = np.mean(np.abs(range_errors)) * 1000
        range_rmse = np.sqrt(
            np.mean(range_errors ** 2)
        ) * 1000
    else:
        range_mae = np.nan
        range_rmse = np.nan

    print(
        f"{lower}-{upper:<10}"
        f"{total_pixels:<10}"
        f"{range_invalid_rate:<14.2f}"
        f"{range_mae:<12.2f}"
        f"{range_rmse:<12.2f}"
    )


# --------------------------------------------------
# 6. Visualise depth corruption
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vmin = np.nanpercentile(depth[valid_original], 1)
vmax = np.nanpercentile(depth[valid_original], 99)

axes[0].imshow(
    np.where(valid_original, depth, np.nan),
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("Ideal depth")

axes[1].imshow(
    noisy_depth,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("Simulated D455 depth")

axes[2].imshow(newly_invalid, cmap="gray")
axes[2].set_title("Newly invalid pixels")

for ax in axes:
    ax.axis("off")


# --------------------------------------------------
# 7. Save a versioned figure
# --------------------------------------------------

os.makedirs("report/figures", exist_ok=True)

existing = glob.glob(
    "report/figures/depth_corruption_comparison_*.png"
)

numbers = [
    int(match.group(1))
    for filename in existing
    if (match := re.search(r"comparison_(\d+)\.png$", filename))
]

figure_number = max(numbers, default=0) + 1

filename = (
    f"report/figures/depth_corruption_comparison_"
    f"{figure_number:03d}.png"
)

plt.tight_layout()

plt.savefig(
    filename,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"\nFigure saved to: {filename}")
