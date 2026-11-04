import glob
import json
import os

import h5py
import numpy as np


ROOT = "outputs/randomized_pose_dataset"
TARGET_CATEGORY = 4

# Target extrusion dimensions from the scene generator.
HALF_EXTENTS = np.array([0.65, 0.04, 0.04])

# OpenCV camera coordinates -> Blender camera coordinates.
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def get_intrinsics(intrinsics):
    """Support either a 3x3 matrix or a dictionary."""
    if isinstance(intrinsics, dict):
        if "K" in intrinsics:
            return np.asarray(intrinsics["K"], dtype=float)
        if "matrix" in intrinsics:
            return np.asarray(intrinsics["matrix"], dtype=float)

        return np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1]
        ], dtype=float)

    return np.asarray(intrinsics, dtype=float)


def backproject_to_world(depth, mask, K, camera_to_world):
    """Back-project selected depth pixels into Blender world coordinates."""
    v, u = np.where(mask)
    z = depth[v, u]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points_cv = np.column_stack([
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
        np.ones_like(z)
    ])

    points_blender_camera = (CV_TO_BLENDER @ points_cv.T).T
    points_world = (camera_to_world @ points_blender_camera.T).T

    return points_world[:, :3]


def box_surface_distance(points, half_extents):
    """
    Approximate distance to the surface of an axis-aligned box.

    Points outside the box are measured against its boundary.
    Points inside are measured against their nearest face.
    """
    a = np.abs(points)
    outside = np.maximum(a - half_extents, 0)
    outside_distance = np.linalg.norm(outside, axis=1)

    inside_distance = np.min(
        np.maximum(half_extents - a, 0),
        axis=1
    )

    inside = np.all(a <= half_extents, axis=1)

    return np.where(inside, inside_distance, outside_distance)


def main():
    with open(f"{ROOT}/pose_labels.json") as f:
        meta = json.load(f)

    K = get_intrinsics(meta["intrinsics"])

    files = sorted(
        glob.glob(os.path.join(ROOT, "*.hdf5")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )

    if len(files) != len(meta["samples"]):
        raise RuntimeError(
            f"File/label mismatch: {len(files)} files, "
            f"{len(meta['samples'])} labels"
        )

    results = []

    for i, path in enumerate(files):
        sample = meta["samples"][i]

        T_target = np.asarray(sample["target_to_world"], dtype=float)
        T_camera = np.asarray(
            sample["camera_to_world_blender"], dtype=float
        )

        with h5py.File(path, "r") as f:
            seg = np.squeeze(f["category_id_segmaps"][:])
            depth = np.squeeze(f["depth"][:])

        mask = (
            (seg == TARGET_CATEGORY)
            & np.isfinite(depth)
            & (depth > 0)
            & (depth < 10)
        )

        points_world = backproject_to_world(
            depth, mask, K, T_camera
        )

        # Transform observed points into the labelled target frame.
        points_h = np.column_stack([
            points_world,
            np.ones(len(points_world))
        ])

        points_local = (
            np.linalg.inv(T_target) @ points_h.T
        ).T[:, :3]

        distances = box_surface_distance(
            points_local, HALF_EXTENTS
        )

        outside = np.any(
            np.abs(points_local) > HALF_EXTENTS + 0.005,
            axis=1
        )

        results.append({
            "sample": i,
            "points": len(points_local),
            "median_surface_error_mm": np.median(distances) * 1000,
            "p95_surface_error_mm": np.percentile(distances, 95) * 1000,
            "outside_5mm_fraction": np.mean(outside),
            "local_centroid": points_local.mean(axis=0)
        })

    median_errors = [
        r["median_surface_error_mm"] for r in results
    ]
    p95_errors = [
        r["p95_surface_error_mm"] for r in results
    ]
    outside_fractions = [
        r["outside_5mm_fraction"] for r in results
    ]

    print("\nGEOMETRY-TO-LABEL VALIDATION")
    print("============================")
    print("Samples:", len(results))
    print("Median surface error (mm):")
    print("  Median:", np.median(median_errors))
    print("  P95:", np.percentile(median_errors, 95))
    print("  Maximum:", np.max(median_errors))

    print("\n95th-percentile surface error across samples (mm):")
    print("  Median:", np.median(p95_errors))
    print("  Maximum:", np.max(p95_errors))

    print("\nFraction of points outside labelled box by >5 mm:")
    print("  Median:", np.median(outside_fractions))
    print("  Maximum:", np.max(outside_fractions))

    print("\nFirst five samples:")
    for r in results[:5]:
        print(
            f"Sample {r['sample']:3d}: "
            f"points={r['points']:4d}, "
            f"median_surface_error={r['median_surface_error_mm']:.3f} mm, "
            f"P95={r['p95_surface_error_mm']:.3f} mm, "
            f"outside={100*r['outside_5mm_fraction']:.2f}%"
        )

    worst = max(
        results,
        key=lambda r: r["median_surface_error_mm"]
    )
    print("\nWorst sample by median surface error:")
    print(worst)


if __name__ == "__main__":
    main()