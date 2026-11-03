import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.sensor.depth_noise import apply_depth_noise

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"

OUTPUT_DIR = "report/figures"

CATEGORY_ID = 2
SEED = 42

# BlenderProc synthetic camera intrinsics
FX = 888.88909234
FY = 888.88909234
CX = 319.5
CY = 239.5

# Plotting settings
MAX_POINTS = 12000
RANDOM_SEED = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load scene data
# ============================================================

print("Loading manufacturing scene...")

with h5py.File(INPUT_FILE, "r") as f:
    depth = f["depth"][()]
    segmentation = f["category_id_segmaps"][()]
    rgb = f["colors"][()]

print("Depth:", depth.shape)
print("Segmentation:", segmentation.shape)
print("RGB:", rgb.shape)


# ============================================================
# Generate corrupted depth
# ============================================================

print("Applying simulated depth corruption...")

corrupted_depth, invalid_mask, edge_mask = apply_depth_noise(
    depth,
    segmentation,
    seed=SEED
)


# ============================================================
# Point-cloud reconstruction
# ============================================================

def reconstruct_point_cloud(depth_map, mask):
    """Back-project valid depth pixels into camera coordinates."""

    v, u = np.where(mask)

    z = depth_map[v, u]

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    points = np.column_stack((x, y, z))

    return points


profile_mask = (
    (segmentation == CATEGORY_ID)
    & np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

ideal_points = reconstruct_point_cloud(
    depth,
    profile_mask
)

corrupted_mask = (
    profile_mask
    & np.isfinite(corrupted_depth)
    & (corrupted_depth > 0)
)

corrupted_points = reconstruct_point_cloud(
    corrupted_depth,
    corrupted_mask
)

ideal_centroid = ideal_points.mean(axis=0)
corrupted_centroid = corrupted_points.mean(axis=0)

displacement = corrupted_centroid - ideal_centroid
displacement_norm = np.linalg.norm(displacement) * 1000

print("\n========== RESULTS ==========")
print("Ideal points:", len(ideal_points))
print("Corrupted points:", len(corrupted_points))
print("Completeness:", len(corrupted_points) / len(ideal_points) * 100, "%")
print("Ideal centroid:", ideal_centroid)
print("Corrupted centroid:", corrupted_centroid)
print("Displacement:", displacement_norm, "mm")


# ============================================================
# Common plotting utilities
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

# Use the same sampled ideal points in the full-view plots.
# All points are still used for centroid calculations.
if len(ideal_points) > MAX_POINTS:
    ideal_indices = rng.choice(
        len(ideal_points),
        MAX_POINTS,
        replace=False
    )
else:
    ideal_indices = np.arange(len(ideal_points))

ideal_display = ideal_points[ideal_indices]

# Sample corrupted points independently for display.
if len(corrupted_points) > MAX_POINTS:
    corrupted_indices = rng.choice(
        len(corrupted_points),
        MAX_POINTS,
        replace=False
    )
else:
    corrupted_indices = np.arange(len(corrupted_points))

corrupted_display = corrupted_points[corrupted_indices]

# Identical axis limits for both full point-cloud views.
all_display = np.vstack((ideal_display, corrupted_display))

axis_min = all_display.min(axis=0)
axis_max = all_display.max(axis=0)

centre = (axis_min + axis_max) / 2
half_range = (axis_max - axis_min).max() / 2

# A little margin around the object
half_range *= 1.08

def set_equal_view(ax):
    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=25, azim=-60)


# ============================================================
# Figure 1: Ideal versus corrupted point clouds
# ============================================================

print("\nCreating point-cloud comparison...")

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

ax1.scatter(
    ideal_display[:, 0],
    ideal_display[:, 1],
    ideal_display[:, 2],
    s=1,
    alpha=0.7
)

ax2.scatter(
    corrupted_display[:, 0],
    corrupted_display[:, 1],
    corrupted_display[:, 2],
    s=1,
    alpha=0.7
)

ax1.set_title("Ideal depth")
ax2.set_title("Corrupted depth")

set_equal_view(ax1)
set_equal_view(ax2)

fig.suptitle(
    "Effect of simulated depth corruption on 3D reconstruction",
    fontsize=14
)

plt.tight_layout()

figure1_path = os.path.join(
    OUTPUT_DIR,
    "pointcloud_ideal_vs_corrupted_001.png"
)

plt.savefig(figure1_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", figure1_path)


# ============================================================
# Figure 2: Centroid displacement
# ============================================================

print("Creating centroid displacement figure...")

fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Full object view
ax1.scatter(
    ideal_display[:, 0],
    ideal_display[:, 1],
    ideal_display[:, 2],
    s=1,
    alpha=0.25
)

ax1.scatter(
    *ideal_centroid,
    marker="o",
    s=90,
    label="Ideal centroid"
)

ax1.scatter(
    *corrupted_centroid,
    marker="^",
    s=90,
    label="Corrupted centroid"
)

ax1.plot(
    [ideal_centroid[0], corrupted_centroid[0]],
    [ideal_centroid[1], corrupted_centroid[1]],
    [ideal_centroid[2], corrupted_centroid[2]],
    linewidth=2,
    label="Centroid displacement"
)

ax1.set_title("Centroids on reconstructed object")
ax1.legend()

set_equal_view(ax1)

# Zoomed view: use all three coordinates and equal scaling.
midpoint = (ideal_centroid + corrupted_centroid) / 2
zoom_radius = max(np.max(np.abs(displacement)) * 8, 0.002)

ax2.scatter(
    *ideal_centroid,
    marker="o",
    s=110,
    label="Ideal centroid"
)

ax2.scatter(
    *corrupted_centroid,
    marker="^",
    s=110,
    label="Corrupted centroid"
)

ax2.plot(
    [ideal_centroid[0], corrupted_centroid[0]],
    [ideal_centroid[1], corrupted_centroid[1]],
    [ideal_centroid[2], corrupted_centroid[2]],
    linewidth=2
)

ax2.set_xlim(midpoint[0] - zoom_radius, midpoint[0] + zoom_radius)
ax2.set_ylim(midpoint[1] - zoom_radius, midpoint[1] + zoom_radius)
ax2.set_zlim(midpoint[2] - zoom_radius, midpoint[2] + zoom_radius)

ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.set_title(
    f"Centroid displacement: {displacement_norm:.2f} mm"
)
ax2.legend()
ax2.view_init(elev=25, azim=-60)

fig.suptitle(
    "Effect of depth corruption on the estimated 3D centroid",
    fontsize=14
)

plt.tight_layout()

figure2_path = os.path.join(
    OUTPUT_DIR,
    "centroid_displacement_001.png"
)

plt.savefig(figure2_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", figure2_path)


# ============================================================
# Animation: synchronized rotating point clouds
# ============================================================

print("Creating rotating point-cloud animation...")

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

ax1.scatter(
    ideal_display[:, 0],
    ideal_display[:, 1],
    ideal_display[:, 2],
    s=1,
    alpha=0.7
)

ax2.scatter(
    corrupted_display[:, 0],
    corrupted_display[:, 1],
    corrupted_display[:, 2],
    s=1,
    alpha=0.7
)

ax1.set_title("Ideal depth")
ax2.set_title("Corrupted depth")

set_equal_view(ax1)
set_equal_view(ax2)

fig.suptitle(
    "3D reconstruction under simulated depth corruption"
)

def update(frame):
    angle = -60 + frame * 3

    ax1.view_init(elev=25, azim=angle)
    ax2.view_init(elev=25, azim=angle)

    return []

animation = FuncAnimation(
    fig,
    update,
    frames=120,
    interval=80,
    blit=False
)

animation_path = os.path.join(
    OUTPUT_DIR,
    "pointcloud_comparison_animation.gif"
)

animation.save(
    animation_path,
    writer=PillowWriter(fps=12)
)

plt.close(fig)

print("Saved:", animation_path)

print("\nAll visualisations completed.")