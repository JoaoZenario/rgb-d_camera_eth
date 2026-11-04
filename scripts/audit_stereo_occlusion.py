import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
GROUND_TRUTH_FILE = "outputs/manufacturing_scene/ground_truth.json"

TARGET_CATEGORY = 4
BASELINE_M = 0.095

# Small tolerance to avoid classifying tiny numerical differences
# and rasterisation collisions as occlusions.
OCCLUSION_TOLERANCE_M = 0.002


def load_data():
    with h5py.File(INPUT_FILE, "r") as f:
        depth = np.squeeze(f["depth"][...]).astype(np.float32)
        segmentation = np.squeeze(
            f["category_id_segmaps"][...]
        )

    with open(GROUND_TRUTH_FILE, "r") as f:
        ground_truth = json.load(f)

    return depth, segmentation, ground_truth


def backproject_blender_camera(depth, K):
    """
    Back-project a BlenderProc depth image.

    Blender camera convention:
      +X = right
      +Y = up
      -Z = forward

    The depth values are treated as camera-axis depth.
    """
    height, width = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    vv, uu = np.indices((height, width))

    z = depth.astype(np.float64)

    x = (uu - cx) * z / fx
    y = -(vv - cy) * z / fy
    z_blender = -z

    points = np.stack(
        [x, y, z_blender, np.ones_like(z)],
        axis=-1,
    )

    return points.reshape(-1, 4)


def simulate_right_view(
    depth,
    segmentation,
    K,
    camera_to_world,
    baseline_sign,
):
    """
    Forward-project left-camera visible points into a hypothetical
    right camera displaced along the left camera's local X axis.

    A z-buffer identifies points that are behind another projected
    surface in the right view.

    This is a geometric approximation, not a stereo matcher.
    """
    height, width = depth.shape

    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    target = (segmentation == TARGET_CATEGORY) & valid

    # Only project pixels with valid depth measurements.
    valid_source_indices = np.flatnonzero(valid.ravel())

    points_left = backproject_blender_camera(depth, K)
    points_left = points_left[valid.ravel()]

    # Left camera -> world.
    points_world = (
        camera_to_world @ points_left.T
    ).T

    # Move the right camera along the left camera's local X axis.
    right_camera_to_world = (
        camera_to_world
        @ np.array([
            [1.0, 0.0, 0.0, baseline_sign * BASELINE_M],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )

    world_to_right_camera = np.linalg.inv(
        right_camera_to_world
    )

    # World -> right camera.
    points_right = (
        world_to_right_camera @ points_world.T
    ).T

    x = points_right[:, 0]
    y = points_right[:, 1]
    z_blender = points_right[:, 2]

    # Positive forward depth in Blender's -Z camera convention.
    z_right = -z_blender

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u_right = fx * x / z_right + cx
        v_right = cy - fy * y / z_right

    # Nearest-pixel forward projection.
    u_pixel = np.rint(u_right).astype(np.int64)
    v_pixel = np.rint(v_right).astype(np.int64)

    in_bounds = (
        (z_right > 0)
        & np.isfinite(z_right)
        & (u_pixel >= 0)
        & (u_pixel < width)
        & (v_pixel >= 0)
        & (v_pixel < height)
    )

    # Map compact point-array indices back to original image pixels.
    source_indices = valid_source_indices[
        np.flatnonzero(in_bounds)
    ]

    destination_indices = (
        v_pixel[in_bounds] * width
        + u_pixel[in_bounds]
    )

    projected_depth = z_right[in_bounds]

    # Build a right-view z-buffer.
    # Category of each source point, retained for occluder attribution.
    source_categories = segmentation.ravel()[source_indices]

    # Sort projected points by destination pixel, then by depth.
    # The first point at each destination pixel is the nearest.
    order = np.lexsort((
        projected_depth,
        destination_indices,
    ))

    sorted_destinations = destination_indices[order]

    first_in_pixel = np.r_[
        True,
        sorted_destinations[1:] != sorted_destinations[:-1],
    ]

    nearest_order_indices = order[first_in_pixel]

    nearest_destinations = destination_indices[
        nearest_order_indices
    ]

    nearest_depths = projected_depth[
        nearest_order_indices
    ]

    nearest_categories = source_categories[
        nearest_order_indices
    ]

    # For every projected point, find the nearest point at the
    # same right-image pixel.
    group_indices = np.searchsorted(
        nearest_destinations,
        destination_indices,
    )

    nearest_depth = nearest_depths[group_indices]
    nearest_category = nearest_categories[group_indices]

    # A point is behind another point if its projected depth is
    # sufficiently greater than the nearest depth.
    occluded = (
        projected_depth
        > nearest_depth + OCCLUSION_TOLERANCE_M
    )

    # Distinguish cross-category occlusion from same-category
    # projection collisions.
    cross_category_occlusion = (
        occluded
        & (source_categories != nearest_category)
    )

    # Map results back to the original left-image pixels.
    source_occluded = np.zeros(
        height * width,
        dtype=bool,
    )

    source_cross_category_occluded = np.zeros(
        height * width,
        dtype=bool,
    )

    source_occluded[source_indices[occluded]] = True

    source_cross_category_occluded[
        source_indices[cross_category_occlusion]
    ] = True

    source_occluded = source_occluded.reshape(
        height, width
    )

    source_cross_category_occluded = (
        source_cross_category_occluded.reshape(
            height, width
        )
    )

    target_occluded = source_occluded & target

    target_cross_category_occluded = (
        source_cross_category_occluded & target
    )

    return {
        "valid": valid,
        "target": target,
        "source_occluded": source_occluded,
        "target_occluded": target_occluded,
        "projected_count": len(source_indices),
        "occluded_count": int(np.count_nonzero(occluded)),
        "target_count": int(np.count_nonzero(target)),
        "target_occluded_count": int(np.count_nonzero(target_occluded)),
        "source_cross_category_occluded":
            source_cross_category_occluded,
        "target_cross_category_occluded":
            target_cross_category_occluded,
        "cross_category_occluded_count": int(
            np.count_nonzero(cross_category_occlusion)
        ),
        "target_cross_category_occluded_count": int(
            np.count_nonzero(target_cross_category_occluded)
        ),
    }


def main():
    depth, segmentation, gt = load_data()

    K = np.asarray(
        gt["camera"]["intrinsics"],
        dtype=np.float64,
    )

    camera_to_world = np.asarray(
        gt["camera"]["camera_to_world_blender"],
        dtype=np.float64,
    )

    print("\n========== STEREO OCCLUSION AUDIT ==========")
    print(f"Image: {depth.shape[1]} x {depth.shape[0]}")
    print(f"Nominal baseline: {BASELINE_M * 1000:.1f} mm")
    print(
        "Occlusion tolerance:",
        f"{OCCLUSION_TOLERANCE_M * 1000:.1f} mm",
    )

    results = {}

    for sign in (-1, 1):
        name = "minus_X" if sign < 0 else "plus_X"

        result = simulate_right_view(
            depth,
            segmentation,
            K,
            camera_to_world,
            sign,
        )

        results[name] = result

        target_count = result["target_count"]
        target_occluded = result["target_occluded_count"]
        target_cross = result["target_cross_category_occluded_count"]

        print(f"\n--- Baseline direction: {name} ---")
        print(
            "Valid target pixels:",
            target_count,
        )
        print(
            "Target points projected into right image:",
            result["projected_count"],
        )
        print(
            "All projected points classified occluded:",
            result["occluded_count"],
        )
        print(
            "Occluded target pixels:",
            target_occluded,
        )
        print(
            "Target pixels occluded by a different category:",
            target_cross,
        )

        print(
            "Cross-category target occlusion fraction:",
            f"{100 * target_cross / target_count:.2f}%",
        )

        print(
            "All cross-category occlusion events:",
            result["cross_category_occluded_count"],
        )
        print(
            "Target occlusion fraction:",
            f"{100 * target_occluded / target_count:.2f}%",
        )

    # Visualise both directional occlusion masks.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        segmentation,
        cmap="tab20",
        interpolation="nearest",
    )
    axes[0].set_title("Semantic segmentation")

    for ax, name in zip(
        axes[1:],
        ("minus_X", "plus_X"),
    ):
        target = results[name]["target"]
        occluded = results[name]["target_occluded"]

        ax.imshow(target, cmap="gray")
        ax.imshow(
            np.ma.masked_where(
                ~occluded,
                occluded,
            ),
            cmap="Reds",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        fraction = (
            100
            * results[name]["target_occluded_count"]
            / results[name]["target_count"]
        )

        ax.set_title(
            f"{name}: target occlusion {fraction:.2f}%"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        "report/figures/stereo_occlusion_audit.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()