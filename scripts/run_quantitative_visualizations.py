import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.sensor.depth_noise import apply_depth_noise

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
OUTPUT_DIR = "report/figures"

CATEGORY_ID = 2
SEED = 42

# Synthetic BlenderProc camera intrinsics
FX = 888.88909234
FY = 888.88909234
CX = 319.5
CY = 239.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load scene
# ============================================================

print("Loading scene...")

with h5py.File(INPUT_FILE, "r") as f:
    depth = f["depth"][()]
    segmentation = f["category_id_segmaps"][()]
    rgb = f["colors"][()]

if depth.shape != segmentation.shape:
    raise ValueError("Depth and segmentation shapes do not match.")

print("Depth shape:", depth.shape)
print("Segmentation shape:", segmentation.shape)


# ============================================================
# Helpers
# ============================================================

def valid_depth(d):
    return np.isfinite(d) & (d > 0) & (d < 1000)


def calculate_depth_metrics(ideal, measured, mask):
    """
    Calculate metrics over the supplied region.

    Error metrics use only pixels valid in both maps.
    Invalid rate uses all originally valid pixels in the region.
    """
    original_valid = mask & valid_depth(ideal)
    measured_valid = mask & valid_depth(measured)

    n_original = int(original_valid.sum())
    n_measured = int(measured_valid.sum())

    if n_original == 0:
        return None

    common = original_valid & measured_valid
    errors = measured[common] - ideal[common]

    if len(errors) > 0:
        mae = np.mean(np.abs(errors)) * 1000
        rmse = np.sqrt(np.mean(errors ** 2)) * 1000
        bias = np.mean(errors) * 1000
    else:
        mae = np.nan
        rmse = np.nan
        bias = np.nan

    invalid_rate = (n_original - n_measured) / n_original * 100

    return {
        "n": n_original,
        "remaining": n_measured,
        "invalid_rate": invalid_rate,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def reconstruct_point_cloud(depth_map, mask):
    v, u = np.where(mask)
    z = depth_map[v, u]

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    return np.column_stack((x, y, z))


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# ============================================================
# Generate all corruption configurations
# ============================================================

print("\nGenerating corruption configurations...")

# Gaussian only
depth_gaussian, _, _ = apply_depth_noise(
    depth,
    segmentation,
    seed=SEED,
    gaussian_noise=True,
    range_missing=False,
    boundary_failures=False,
)

# Range-dependent missing measurements only
depth_range, _, _ = apply_depth_noise(
    depth,
    segmentation,
    seed=SEED,
    gaussian_noise=False,
    range_missing=True,
    boundary_failures=False,
)

# Boundary failures only
depth_boundary, _, _ = apply_depth_noise(
    depth,
    segmentation,
    seed=SEED,
    gaussian_noise=False,
    range_missing=False,
    boundary_failures=True,
)

# Full corruption model
depth_full, _, edge_mask = apply_depth_noise(
    depth,
    segmentation,
    seed=SEED,
    gaussian_noise=True,
    range_missing=True,
    boundary_failures=True,
)

ideal_valid = valid_depth(depth)


# ============================================================
# Figure 1: Depth error versus distance
# ============================================================

print("\nCalculating range-dependent metrics...")

# Use the same distance intervals as the report.
range_bins = [
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
]

range_labels = ["2–3 m", "3–4 m", "4–5 m"]

range_results = []

for low, high in range_bins:
    region = (
        ideal_valid
        & (depth >= low)
        & (depth < high)
    )

    metrics = calculate_depth_metrics(depth, depth_full, region)
    range_results.append(metrics)

    if metrics is None:
        print(f"{low}-{high} m: no valid samples")
    else:
        print(
            f"{low}-{high} m | "
            f"N={metrics['n']} | "
            f"Invalid={metrics['invalid_rate']:.2f}% | "
            f"MAE={metrics['mae']:.2f} mm | "
            f"RMSE={metrics['rmse']:.2f} mm"
        )

fig, axes = plt.subplots(
    2, 1,
    figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

x = np.arange(len(range_labels))
width = 0.34

mae_values = [m["mae"] for m in range_results]
rmse_values = [m["rmse"] for m in range_results]
invalid_values = [m["invalid_rate"] for m in range_results]
sample_counts = [m["n"] for m in range_results]

# Highlight the sparse 4–5 m interval in both panels.
for ax in axes:
    ax.axvspan(
        1.5, 2.5,
        color="grey",
        alpha=0.12,
        zorder=0
    )

mae_bars = axes[0].bar(
    x - width / 2,
    mae_values,
    width,
    label="MAE"
)

rmse_bars = axes[0].bar(
    x + width / 2,
    rmse_values,
    width,
    label="RMSE"
)

# Hatch the sparse interval to distinguish it visually.
mae_bars[2].set_hatch("//")
rmse_bars[2].set_hatch("//")

axes[0].set_ylabel("Depth error (mm)")
axes[0].set_title("Depth error as a function of distance")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.25)

for i in range(len(x)):
    height = max(mae_values[i], rmse_values[i])
    axes[0].text(
        i,
        height + 0.8,
        f"{height:.2f} mm",
        ha="center",
        fontsize=8
    )

invalid_bars = axes[1].bar(
    x,
    invalid_values,
    width=0.5
)

invalid_bars[2].set_hatch("//")

axes[1].set_ylabel("Invalid depth (%)")
axes[1].set_xlabel("Distance interval and number of valid reference pixels")
axes[1].set_xticks(x)

# Put sample counts directly beneath each distance interval.
tick_labels = [
    f"{label}\nN = {count:,} pixels"
    + ("\n(SPARSE SAMPLE)" if count < 500 else "")
    for label, count in zip(range_labels, sample_counts)
]

axes[1].set_xticklabels(tick_labels)

for i, value in enumerate(invalid_values):
    axes[1].text(
        i,
        value + 0.15,
        f"{value:.2f}%",
        ha="center",
        fontsize=9
    )

axes[1].set_ylim(
    0,
    max(invalid_values) + 1.5
)

fig.suptitle(
    "Range-dependent depth error and measurement availability"
)

fig.tight_layout()

save_figure(
    fig,
    "depth_error_vs_range_001.png"
)


# ============================================================
# Figure 2: Ablation study
# ============================================================

print("\nCalculating ablation metrics...")

ablation_configs = [
    ("Baseline", depth),
    ("Gaussian", depth_gaussian),
    ("Range missing", depth_range),
    ("Boundary failures", depth_boundary),
    ("Full model", depth_full),
]

ablation_results = []

for name, measured in ablation_configs:
    metrics = calculate_depth_metrics(
        depth,
        measured,
        ideal_valid
    )

    ablation_results.append(metrics)

    print(
        f"{name:20s} | "
        f"Invalid={metrics['invalid_rate']:.2f}% | "
        f"MAE={metrics['mae']:.2f} mm | "
        f"RMSE={metrics['rmse']:.2f} mm"
    )

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

names = [item[0] for item in ablation_configs]
invalid_rates = [m["invalid_rate"] for m in ablation_results]
rmses = [m["rmse"] for m in ablation_results]

x = np.arange(len(names))

# Left: missing-depth rate
availability_bars = axes[0].bar(
    x,
    invalid_rates
)

axes[0].set_ylabel("Invalid depth (%)")
axes[0].set_title("Measurement availability")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=25, ha="right")
axes[0].grid(axis="y", alpha=0.25)

# Right: numerical error on surviving valid measurements
rmse_bars = axes[1].bar(
    x,
    rmses
)

axes[1].set_ylabel("RMSE (mm)")
axes[1].set_title("Numerical depth error")
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, rotation=25, ha="right")
axes[1].grid(axis="y", alpha=0.25)

# Make zero values explicit rather than leaving apparently empty bars.
for ax, bars, values in [
    (axes[0], availability_bars, invalid_rates),
    (axes[1], rmse_bars, rmses),
]:
    max_value = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, max_value * 1.25 + 0.1)

    for bar, value in zip(bars, values):
        if value == 0:
            ax.annotate(
                "0.00",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=8
            )

fig.suptitle(
    "Ablation of simulated depth corruption mechanisms"
)

fig.tight_layout()

save_figure(
    fig,
    "corruption_ablation_quantitative_001.png"
)


# ============================================================
# Figure 3: Centroid displacement components
# ============================================================

print("\nCalculating centroid displacement...")

profile_mask = (
    (segmentation == CATEGORY_ID)
    & ideal_valid
)

ideal_profile_points = reconstruct_point_cloud(
    depth,
    profile_mask
)

corrupted_profile_mask = (
    profile_mask
    & valid_depth(depth_full)
)

corrupted_profile_points = reconstruct_point_cloud(
    depth_full,
    corrupted_profile_mask
)

ideal_centroid = ideal_profile_points.mean(axis=0)
corrupted_centroid = corrupted_profile_points.mean(axis=0)

displacement = corrupted_centroid - ideal_centroid
displacement_mm = displacement * 1000

displacement_norm = np.linalg.norm(displacement_mm)

print("Ideal centroid:", ideal_centroid)
print("Corrupted centroid:", corrupted_centroid)
print("Displacement components (mm):", displacement_mm)
print("Total displacement (mm):", displacement_norm)

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [1.5, 1]}
)

components = ["X", "Y", "Z"]
values = displacement_mm

axes[0].barh(components, values)
axes[0].axvline(0, linewidth=0.8)
axes[0].set_xlabel("Centroid displacement (mm)")
axes[0].set_title("Signed displacement by camera axis")
axes[0].grid(axis="x", alpha=0.25)

for i, value in enumerate(values):
    axes[0].text(
        value + (0.06 if value >= 0 else -0.06),
        i,
        f"{value:+.2f} mm",
        va="center",
        ha="left" if value >= 0 else "right"
    )

axes[1].bar(
    ["Euclidean\n displacement"],
    [displacement_norm]
)

axes[1].set_ylabel("Displacement (mm)")
axes[1].set_title("Total centroid displacement")
axes[1].set_ylim(0, displacement_norm * 1.3)
axes[1].text(
    0,
    displacement_norm + displacement_norm * 0.04,
    f"{displacement_norm:.2f} mm",
    ha="center"
)
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle("Effect of depth corruption on the 3D centroid")
fig.tight_layout()

save_figure(
    fig,
    "centroid_displacement_components_001.png"
)


# ============================================================
# Figure 4: Spatial distribution of missing depth
# ============================================================

print("\nAnalysing spatial failure patterns...")

range_invalid = ideal_valid & ~valid_depth(depth_range)
boundary_invalid = ideal_valid & ~valid_depth(depth_boundary)

overlap_invalid = range_invalid & boundary_invalid

range_only = range_invalid & ~boundary_invalid
boundary_only = boundary_invalid & ~range_invalid

combined_invalid = ideal_valid & ~valid_depth(depth_full)

explained_invalid = range_invalid | boundary_invalid

if not np.array_equal(combined_invalid, explained_invalid):
    raise RuntimeError(
        "The isolated failure masks do not reproduce "
        "the combined invalid mask. Check the noise implementation."
    )

print("Range-only invalid pixels:", int(range_only.sum()))
print("Boundary-only invalid pixels:", int(boundary_only.sum()))
print("Overlapping invalid pixels:", int(overlap_invalid.sum()))
print("Total invalid pixels:", int(combined_invalid.sum()))

# Prepare ideal depth background.
base_depth = np.where(ideal_valid, depth, np.nan)

# Each row shows the same failure pattern in RGB and depth.
panels = [
    ("Ideal reference", None),
    ("Range-only failures", range_only),
    ("Boundary-only failures", boundary_only),
    ("Overlapping failures", overlap_invalid),
    ("Combined invalid measurements", combined_invalid),
]

fig, axes = plt.subplots(
    len(panels),
    2,
    figsize=(12, 17)
)

for row, (title, failure_mask) in enumerate(panels):

    # --------------------------------------------------------
    # RGB view
    # --------------------------------------------------------

    axes[row, 0].imshow(rgb)

    if failure_mask is not None:
        rgb_overlay = np.ma.masked_where(
            ~failure_mask,
            failure_mask.astype(float)
        )

        axes[row, 0].imshow(
            rgb_overlay,
            cmap="autumn",
            alpha=0.9,
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

    axes[row, 0].set_title(
        f"{title} — RGB"
        + (
            f"\n{int(failure_mask.sum()):,} invalid pixels"
            if failure_mask is not None
            else ""
        )
    )

    axes[row, 0].set_xlabel("Image column (pixels)")
    axes[row, 0].set_ylabel("Image row (pixels)")

    # --------------------------------------------------------
    # Ideal depth view with the same failure overlay
    # --------------------------------------------------------

    axes[row, 1].imshow(
        base_depth,
        cmap="viridis"
    )

    if failure_mask is not None:
        depth_overlay = np.ma.masked_where(
            ~failure_mask,
            failure_mask.astype(float)
        )

        axes[row, 1].imshow(
            depth_overlay,
            cmap="autumn",
            alpha=0.9,
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

    axes[row, 1].set_title(
        f"{title} — ideal depth"
        + (
            f"\n{int(failure_mask.sum()):,} invalid pixels"
            if failure_mask is not None
            else ""
        )
    )

    axes[row, 1].set_xlabel("Image column (pixels)")
    axes[row, 1].set_ylabel("Image row (pixels)")

fig.suptitle(
    "Spatial distribution of simulated invalid depth measurements\n"
    "Failure locations shown on RGB and ideal depth images",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

save_figure(
    fig,
    "spatial_missing_depth_001.png"
)

# ============================================================
# Save quantitative summary
# ============================================================

csv_path = os.path.join(
    "report",
    "quantitative_visualization_results.csv"
)

with open(csv_path, "w", encoding="utf-8") as f:
    f.write(
        "distance_range,pixels,invalid_percent,mae_mm,rmse_mm,bias_mm\n"
    )

    for label, metrics in zip(range_labels, range_results):
        f.write(
            f"{label},{metrics['n']},"
            f"{metrics['invalid_rate']:.4f},"
            f"{metrics['mae']:.4f},"
            f"{metrics['rmse']:.4f},"
            f"{metrics['bias']:.4f}\n"
        )

print("\nSaved quantitative CSV:", csv_path)
print("\nAll quantitative visualisations completed.")