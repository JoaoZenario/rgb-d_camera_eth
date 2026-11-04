import os
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.sensor.depth_noise import apply_depth_noise

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"

CSV_FILE = "report/perception_results.csv"
FIGURE_FILE = "report/figures/perception_comparison_001.png"

SEED = 42

# BlenderProc camera intrinsics (pixels)
FX = 888.88909234
FY = 888.88909234
CX = 319.5
CY = 239.5

# Semantic category ID for aluminium profiles
PROFILE_CATEGORY_ID = 2

# Maximum number of points displayed in each 3D plot
MAX_DISPLAY_POINTS = 10000


# ============================================================
# 2. LOAD SYNTHETIC SCENE
# ============================================================

with h5py.File(INPUT_FILE, "r") as f:
    depth_ideal = np.squeeze(f["depth"][...]).astype(np.float32)

    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

    rgb = np.asarray(f["colors"][...])

print("Scene loaded.")
print("Depth shape:", depth_ideal.shape)
print("Segmentation shape:", segmentation.shape)
print("RGB shape:", rgb.shape)


# ============================================================
# 3. SELECT ALUMINIUM PROFILES
# ============================================================

profile_mask = segmentation == PROFILE_CATEGORY_ID

valid_ideal = (
    np.isfinite(depth_ideal)
    & (depth_ideal > 0)
    & (depth_ideal < 1000)
)

# Only profile pixels with valid depth are used.
ideal_mask = profile_mask & valid_ideal

print("\n========== OBJECT SELECTION ==========")
print("Profile pixels:", np.count_nonzero(profile_mask))
print("Valid profile depth pixels:", np.count_nonzero(ideal_mask))


# ============================================================
# 4. GENERATE CORRUPTED DEPTH
# ============================================================

depth_corrupted, invalid_mask, edge_mask = apply_depth_noise(
    depth_ideal,
    segmentation,
    seed=SEED,
    gaussian_noise=True,
    range_missing=True,
    boundary_failures=True,
)

valid_corrupted = (
    np.isfinite(depth_corrupted)
    & (depth_corrupted > 0)
    & (depth_corrupted < 1000)
)

corrupted_mask = profile_mask & valid_corrupted

print("\n========== DEPTH CORRUPTION ==========")
print("Valid ideal profile pixels:", np.count_nonzero(ideal_mask))
print("Valid corrupted profile pixels:", np.count_nonzero(corrupted_mask))


# ============================================================
# 5. BACK-PROJECT DEPTH PIXELS INTO 3D
# ============================================================

def depth_to_point_cloud(depth, mask):
    """
    Back-project depth pixels into a 3D point cloud.

    Coordinate convention:
        X: right
        Y: down
        Z: forward (camera optical axis)

    Units: metres.
    """

    v, u = np.where(mask)

    z = depth[v, u]

    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    points = np.column_stack((x, y, z))

    return points.astype(np.float32)


points_ideal = depth_to_point_cloud(
    depth_ideal,
    ideal_mask,
)

points_corrupted = depth_to_point_cloud(
    depth_corrupted,
    corrupted_mask,
)

print("\n========== POINT CLOUDS ==========")
print("Ideal point cloud:", points_ideal.shape)
print("Corrupted point cloud:", points_corrupted.shape)


# ============================================================
# 6. POINT CLOUD COMPLETENESS
# ============================================================

ideal_point_count = len(points_ideal)
corrupted_point_count = len(points_corrupted)

if ideal_point_count == 0:
    raise ValueError("No valid ideal profile points found.")

completeness = (
    corrupted_point_count / ideal_point_count * 100
)


# ============================================================
# 7. CENTROID AND GRASP-POINT PROXY
# ============================================================

centroid_ideal = np.mean(points_ideal, axis=0)
centroid_corrupted = np.mean(points_corrupted, axis=0)

centroid_displacement = (
    centroid_corrupted - centroid_ideal
)

centroid_error = np.linalg.norm(
    centroid_displacement
)

# The visible point-cloud centroid is used as a simple
# geometric grasp-point proxy in this experiment.
grasp_point_ideal = centroid_ideal
grasp_point_corrupted = centroid_corrupted

grasp_displacement = np.linalg.norm(
    grasp_point_corrupted - grasp_point_ideal
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n========== PERCEPTION RESULTS ==========")

print(f"Ideal point count: {ideal_point_count}")
print(f"Corrupted point count: {corrupted_point_count}")

print(f"Point-cloud completeness: {completeness:.2f}%")

print("\nIdeal centroid (m):")
print(centroid_ideal)

print("\nCorrupted centroid (m):")
print(centroid_corrupted)

print("\nCentroid displacement:")
print(f"X: {centroid_displacement[0] * 1000:.2f} mm")
print(f"Y: {centroid_displacement[1] * 1000:.2f} mm")
print(f"Z: {centroid_displacement[2] * 1000:.2f} mm")

print(
    f"\n3D centroid / grasp-point displacement: "
    f"{centroid_error * 1000:.2f} mm"
)


# ============================================================
# 9. SAVE RESULTS TO CSV
# ============================================================

os.makedirs("report", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

results = {
    "Ideal profile points": ideal_point_count,
    "Corrupted profile points": corrupted_point_count,
    "Point cloud completeness (%)": completeness,
    "Centroid displacement X (mm)": centroid_displacement[0] * 1000,
    "Centroid displacement Y (mm)": centroid_displacement[1] * 1000,
    "Centroid displacement Z (mm)": centroid_displacement[2] * 1000,
    "3D grasp proxy displacement (mm)": grasp_displacement * 1000,
}

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])

    for key, value in results.items():
        writer.writerow([key, value])

print(f"\nCSV saved to: {CSV_FILE}")


# ============================================================
# 10. VISUALISE POINT CLOUDS
# ============================================================

def subsample_points(points, max_points, seed=42):
    if len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(points),
        size=max_points,
        replace=False,
    )

    return points[indices]


display_ideal = subsample_points(
    points_ideal, MAX_DISPLAY_POINTS
)

display_corrupted = subsample_points(
    points_corrupted, MAX_DISPLAY_POINTS
)

fig = plt.figure(figsize=(13, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    display_ideal[:, 0],
    display_ideal[:, 1],
    display_ideal[:, 2],
    s=1,
)
ax1.scatter(
    centroid_ideal[0],
    centroid_ideal[1],
    centroid_ideal[2],
    marker="x",
    s=80,
)
ax1.set_title("Ideal depth")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(
    display_corrupted[:, 0],
    display_corrupted[:, 1],
    display_corrupted[:, 2],
    s=1,
)
ax2.scatter(
    centroid_corrupted[0],
    centroid_corrupted[1],
    centroid_corrupted[2],
    marker="x",
    s=80,
)
ax2.set_title("Corrupted depth")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")

plt.suptitle(
    f"Aluminium profile point clouds "
    f"(completeness: {completeness:.2f}%)"
)

plt.tight_layout()
plt.savefig(
    FIGURE_FILE,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print(f"Figure saved to: {FIGURE_FILE}")
print("\nPerception experiment completed.")