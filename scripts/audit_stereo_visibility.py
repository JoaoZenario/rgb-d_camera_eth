import os
import json
import csv

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "outputs/stereo_geometry"
REPORT_DIR = "report"
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

VIEW_FILES = {
    "plus_x": "1.hdf5",
    "minus_x": "2.hdf5",
}

TARGET_CATEGORY = 4

# Primary comparison tolerance, with sensitivity analysis.
TOLERANCES_M = [0.002, 0.005, 0.010]
PRIMARY_TOLERANCE_M = 0.005


# ============================================================
# LOAD DATA
# ============================================================

def load_view(filename):
    path = os.path.join(DATA_DIR, filename)

    with h5py.File(path, "r") as f:
        depth = f["depth"][()]
        segmentation = f["category_id_segmaps"][()]

    return depth.astype(np.float64), segmentation.astype(np.int32)


with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
    metadata = json.load(f)

K = np.asarray(metadata["camera"]["intrinsics"], dtype=np.float64)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

poses = {
    name: np.asarray(pose, dtype=np.float64)
    for name, pose in metadata["camera"]["camera_poses"].items()
}

depth_original, seg_original = load_view("0.hdf5")

height, width = depth_original.shape

valid_source = (
    np.isfinite(depth_original)
    & (depth_original > 0)
    & (depth_original < 1000)
)

source_rows, source_cols = np.nonzero(valid_source)

source_depth = depth_original[valid_source]
source_categories = seg_original[valid_source]

print("Original image:", width, "x", height)
print("Valid source pixels:", len(source_depth))
print(
    "Valid target pixels:",
    int(np.sum(source_categories == TARGET_CATEGORY))
)


# ============================================================
# BACK-PROJECT ORIGINAL IMAGE INTO WORLD COORDINATES
# ============================================================

# OpenCV camera convention:
#   +X right
#   +Y down
#   +Z forward
#
# Blender camera convention:
#   +X right
#   +Y up
#   -Z forward
#
# BlenderProc depth output is axial camera depth, not
# Euclidean distance along the viewing ray.

x_cv = (source_cols - cx) * source_depth / fx
y_cv = (source_rows - cy) * source_depth / fy

points_camera_blender = np.column_stack([
    x_cv,
    -y_cv,
    -source_depth,
    np.ones_like(source_depth)
])

T_original = poses["original"]

points_world = (
    T_original @ points_camera_blender.T
).T


# ============================================================
# PROJECT INTO A DISPLACED CAMERA AND COMPARE DEPTH
# ============================================================

def classify_visibility(
    destination_depth,
    destination_segmentation,
    destination_pose,
    tolerance_m
):
    T_destination_inv = np.linalg.inv(destination_pose)

    points_destination = (
        T_destination_inv @ points_world.T
    ).T

    x = points_destination[:, 0]
    y = points_destination[:, 1]
    z_blender = points_destination[:, 2]

    predicted_depth = -z_blender

    in_front = (
        np.isfinite(predicted_depth)
        & (predicted_depth > 0)
    )

    projected_u = np.full(len(predicted_depth), np.nan)
    projected_v = np.full(len(predicted_depth), np.nan)

    projected_u[in_front] = (
        fx * x[in_front] / predicted_depth[in_front] + cx
    )

    projected_v[in_front] = (
        cy - fy * y[in_front] / predicted_depth[in_front]
    )

    in_image = (
        in_front
        & (projected_u >= 0)
        & (projected_u < width)
        & (projected_v >= 0)
        & (projected_v < height)
    )

    classification = np.full(
        len(source_depth),
        "outside_fov",
        dtype=object
    )

    source_indices = np.where(in_image)[0]

    center_u = np.rint(projected_u[in_image]).astype(int)
    center_v = np.rint(projected_v[in_image]).astype(int)

    # Remove coordinates that round outside the image.
    inside_rounded = (
        (center_u >= 0)
        & (center_u < width)
        & (center_v >= 0)
        & (center_v < height)
    )

    source_indices = source_indices[inside_rounded]
    center_u = center_u[inside_rounded]
    center_v = center_v[inside_rounded]

    predicted = predicted_depth[source_indices]

    # Sample a 3x3 destination neighbourhood.
    neighbour_depths = np.full(
        (len(source_indices), 9),
        np.nan,
        dtype=np.float64
    )

    neighbour_categories = np.zeros(
        (len(source_indices), 9),
        dtype=np.int32
    )

    column = 0

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            uu = center_u + dx
            vv = center_v + dy

            inside = (
                (uu >= 0)
                & (uu < width)
                & (vv >= 0)
                & (vv < height)
            )

            rows = np.where(inside)[0]

            if len(rows):
                neighbour_depths[rows, column] = (
                    destination_depth[vv[rows], uu[rows]]
                )

                neighbour_categories[rows, column] = (
                    destination_segmentation[vv[rows], uu[rows]]
                )

            column += 1

    valid_neighbour = (
        np.isfinite(neighbour_depths)
        & (neighbour_depths > 0)
        & (neighbour_depths < 1000)
    )

    has_depth = valid_neighbour.any(axis=1)

    difference = neighbour_depths - predicted[:, None]

    consistent = (
        valid_neighbour
        & (np.abs(difference) <= tolerance_m)
    )

    closer = (
        valid_neighbour
        & (difference < -tolerance_m)
    )

    farther = (
        valid_neighbour
        & (difference > tolerance_m)
    )

    has_consistent = consistent.any(axis=1)
    has_closer = closer.any(axis=1)
    has_farther = farther.any(axis=1)

    classification[source_indices[~has_depth]] = (
        "no_destination_depth"
    )

    # A matching neighbour is evidence that the source surface
    # is represented in the destination view.
    visible = has_depth & has_consistent

    classification[source_indices[visible]] = (
        "visible_consistent"
    )

    # Only call it occluded when all valid neighbours are nearer.
    occluded = (
        has_depth
        & ~has_consistent
        & has_closer
        & ~has_farther
    )

    # Mixed nearer/farther surfaces without a matching depth
    # are likely to occur near depth discontinuities.
    ambiguous = (
        has_depth
        & ~has_consistent
        & has_closer
        & has_farther
    )

    farther_only = (
        has_depth
        & ~has_consistent
        & ~has_closer
        & has_farther
    )

    classification[source_indices[ambiguous]] = (
        "ambiguous_boundary"
    )

    classification[source_indices[farther_only]] = (
        "farther_mismatch"
    )

    # Attribute robust occlusions using the nearest valid
    # destination surface in the 3x3 neighbourhood.
    occluded_rows = np.where(occluded)[0]

    if len(occluded_rows):
        depths_for_min = np.where(
            valid_neighbour[occluded_rows],
            neighbour_depths[occluded_rows],
            np.inf
        )

        nearest_columns = np.argmin(
            depths_for_min,
            axis=1
        )

        nearest_categories = neighbour_categories[
            occluded_rows,
            nearest_columns
        ]

        source_categories_occluded = source_categories[
            source_indices[occluded_rows]
        ]

        cross_category = (
            source_categories_occluded != nearest_categories
        )

        classification[
            source_indices[occluded_rows[cross_category]]
        ] = "occluded_cross_category"

        classification[
            source_indices[occluded_rows[~cross_category]]
        ] = "occluded_same_category"

    return classification, None, source_indices

# ============================================================
# SUMMARISE CLASSIFICATIONS
# ============================================================

CLASS_NAMES = [
    "visible_consistent",
    "occluded_same_category",
    "occluded_cross_category",
    "ambiguous_boundary",
    "outside_fov",
    "no_destination_depth",
    "farther_mismatch",
]


def summarise(classification, source_mask):
    selected = classification[source_mask]

    counts = {
        name: int(np.sum(selected == name))
        for name in CLASS_NAMES
    }

    total = len(selected)

    percentages = {
        name: (
            100.0 * counts[name] / total
            if total > 0 else 0.0
        )
        for name in CLASS_NAMES
    }

    return {
        "total_source_points": int(total),
        "counts": counts,
        "percentages": percentages,
    }


# ============================================================
# RUN BOTH BASELINE DIRECTIONS AND TOLERANCE SWEEP
# ============================================================

target_mask = source_categories == TARGET_CATEGORY
all_mask = np.ones(len(source_categories), dtype=bool)

all_results = {}
csv_rows = []

for view_name, filename in VIEW_FILES.items():

    print("\n" + "=" * 65)
    print("DESTINATION VIEW:", view_name)

    destination_depth, destination_seg = load_view(filename)

    all_results[view_name] = {}

    for tolerance in TOLERANCES_M:

        classification, _, _ = classify_visibility(
            destination_depth,
            destination_seg,
            poses[view_name],
            tolerance
        )

        tolerance_key = f"{tolerance * 1000:.0f}_mm"

        target_summary = summarise(
            classification,
            target_mask
        )

        scene_summary = summarise(
            classification,
            all_mask
        )

        all_results[view_name][tolerance_key] = {
            "target": target_summary,
            "whole_scene": scene_summary,
        }

        for subset_name, summary in [
            ("target_category_4", target_summary),
            ("whole_scene", scene_summary),
        ]:
            for class_name in CLASS_NAMES:
                csv_rows.append({
                    "view": view_name,
                    "tolerance_mm": tolerance * 1000,
                    "subset": subset_name,
                    "classification": class_name,
                    "count": summary["counts"][class_name],
                    "percentage": summary["percentages"][class_name],
                })

        if tolerance == PRIMARY_TOLERANCE_M:

            print(
                f"\nTarget category {TARGET_CATEGORY} "
                f"({tolerance * 1000:.0f} mm tolerance)"
            )

            for name in CLASS_NAMES:
                count = target_summary["counts"][name]
                pct = target_summary["percentages"][name]

                print(
                    f"  {name:28s}: "
                    f"{count:5d} ({pct:6.2f}%)"
                )

            print(
                "\nWhole scene "
                f"({tolerance * 1000:.0f} mm tolerance)"
            )

            for name in CLASS_NAMES:
                count = scene_summary["counts"][name]
                pct = scene_summary["percentages"][name]

                print(
                    f"  {name:28s}: "
                    f"{count:6d} ({pct:6.2f}%)"
                )


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

json_path = os.path.join(
    REPORT_DIR,
    "stereo_visibility_results.json"
)

with open(json_path, "w") as f:
    json.dump(all_results, f, indent=4)

csv_path = os.path.join(
    REPORT_DIR,
    "stereo_visibility_results.csv"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "view",
            "tolerance_mm",
            "subset",
            "classification",
            "count",
            "percentage",
        ]
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print("\nSaved:", json_path)
print("Saved:", csv_path)


# ============================================================
# PLOT TARGET VISIBILITY BREAKDOWN
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5.5))

plot_classes = [
    "visible_consistent",
    "occluded_same_category",
    "occluded_cross_category",
    "ambiguous_boundary",
    "outside_fov",
    "no_destination_depth",
    "farther_mismatch",
]

labels = [
    "Visible / consistent",
    "Occluded: same category",
    "Occluded: other category",
    "Ambiguous boundary",
    "Outside field of view",
    "No destination depth",
    "Farther-depth mismatch",
]

x_positions = np.arange(len(VIEW_FILES))
bottom = np.zeros(len(VIEW_FILES))

for class_name, label in zip(plot_classes, labels):

    values = []

    for view_name in VIEW_FILES:
        summary = all_results[view_name][
            f"{PRIMARY_TOLERANCE_M * 1000:.0f}_mm"
        ]["target"]

        values.append(
            summary["percentages"][class_name]
        )

    ax.bar(
        x_positions,
        values,
        bottom=bottom,
        label=label
    )

    bottom += np.asarray(values)

ax.set_xticks(x_positions)
ax.set_xticklabels(["+95 mm", "−95 mm"])

ax.set_ylabel("Percentage of original target pixels (%)")
ax.set_ylim(0, 100)

ax.set_title(
    "Cross-view visibility of the target extrusion\n"
    f"Depth comparison tolerance: "
    f"{PRIMARY_TOLERANCE_M * 1000:.0f} mm"
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False
)

ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

figure_path = os.path.join(
    FIGURE_DIR,
    "stereo_visibility_audit.png"
)

fig.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", figure_path)
print("\nStereo visibility audit completed.")