import json
import h5py
import numpy as np
import open3d as o3d

from pathlib import Path

DATA = Path("outputs/randomized_pose_dataset")
CATEGORY = 4
LENGTH = 1.30  # Known CAD envelope length, metres
SAMPLE_IDS = [0, 19, 126, 189, 198]

CV_TO_BLENDER = np.diag([1., -1., -1., 1.])

def load_sample(sample, K):
    sid = int(sample["sample_id"])

    with h5py.File(DATA / f"{sid}.hdf5", "r") as f:
        depth = np.squeeze(f["depth"][:])
        seg = np.squeeze(f["category_id_segmaps"][:])

    mask = (
        (seg == CATEGORY)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(mask)
    z = depth[v, u].astype(float)

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    T_cw = np.asarray(sample["camera_to_world_blender"])
    points_world = (
        T_cw @ CV_TO_BLENDER @ points_cv.T
    ).T[:, :3]

    return points_world, T_cw[:3, 3]


def estimate_axis(points):
    xy = points[:, :2]
    centred = xy - np.median(xy, axis=0)

    keep = np.ones(len(xy), dtype=bool)

    for _ in range(3):
        if np.count_nonzero(keep) < 3:
            raise ValueError("Too few points for robust axis estimation")

        _, _, vt = np.linalg.svd(
            centred[keep], full_matrices=False
        )
        axis = vt[0]
        normal = np.array([-axis[1], axis[0]])

        residual = np.abs(centred @ normal)
        threshold = max(
            0.012,
            np.percentile(residual[keep], 85)
        )
        keep = residual < threshold

    # Recompute using the final inlier set.
    _, _, vt = np.linalg.svd(
        centred[keep], full_matrices=False
    )
    axis = vt[0]
    axis /= np.linalg.norm(axis)

    return axis


def split_projection_clusters(values, max_gap=0.035):
    """Split sorted 1-D projections at gaps larger than max_gap."""
    order = np.argsort(values)
    sorted_values = values[order]

    cuts = np.where(np.diff(sorted_values) > max_gap)[0] + 1
    groups = np.split(order, cuts)

    return groups


def analyse(
    points,
    camera,
    normal_radius=0.025,
    angle_deg=20,
    min_candidates=20,
    min_cluster_points=None,
):
    if min_cluster_points is None:
        min_cluster_points = min_candidates

    if min_candidates < 1 or min_cluster_points < 1:
        raise ValueError("Candidate thresholds must be positive")

    axis = estimate_axis(points)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=50,
        )
    )

    normals = np.asarray(cloud.normals)
    horizontal = normals[:, :2].copy()
    norm = np.linalg.norm(horizontal, axis=1)

    valid = norm > 0.5
    horizontal[valid] /= norm[valid, None]

    alignment = np.abs(horizontal @ axis)
    candidate_indices = np.where(
        valid & (alignment > np.cos(np.deg2rad(angle_deg)))
    )[0]

    if len(candidate_indices) < min_candidates:
        return axis, len(candidate_indices), []

    # Project candidate points onto the estimated horizontal axis.
    origin = np.median(points, axis=0)
    projection = (
        points[candidate_indices, :2] - origin[:2]
    ) @ axis

    groups = split_projection_clusters(projection, max_gap=0.035)

    planes = []

    for group in groups:
        if len(group) < min_cluster_points:
            continue

        indices = candidate_indices[group]
        cluster = points[indices]

        centroid = cluster.mean(axis=0)

        # Robust plane fit using SVD.
        _, singular_values, vt = np.linalg.svd(
            cluster - centroid,
            full_matrices=False
        )

        plane_normal = vt[-1]
        plane_normal /= np.linalg.norm(plane_normal)

        # Orient the plane normal towards the camera.
        if np.dot(plane_normal, camera - centroid) < 0:
            plane_normal *= -1

        residuals = np.abs(
            (cluster - centroid) @ plane_normal
        )

        # Estimate the two in-plane dimensions of this cluster.
        centred_cluster = cluster - centroid
        _, _, vt_cluster = np.linalg.svd(
            centred_cluster,
            full_matrices=False
        )

        local = centred_cluster @ vt_cluster.T
        dimensions = (
            np.percentile(local, 95, axis=0)
            - np.percentile(local, 5, axis=0)
        )

        dimensions_mm = np.sort(1000 * dimensions)

        planes.append({
            "count": len(cluster),
            "projection": float(np.median(projection[group])),
            "centroid": centroid,
            "normal": plane_normal,
            "median_residual_mm": float(
                1000 * np.median(residuals)
            ),
            "p95_residual_mm": float(
                1000 * np.percentile(residuals, 95)
            ),
            "singular_values": singular_values,
            "dimensions_mm": dimensions_mm,
        })

    planes.sort(key=lambda p: p["projection"])

    return axis, len(candidate_indices), planes


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = {
        int(s["sample_id"]): s
        for s in metadata["samples"]
    }

    for sid in SAMPLE_IDS:
        points, camera = load_sample(samples[sid], K)
        axis, n_candidates, planes = analyse(points, camera)

        T_gt = np.asarray(samples[sid]["target_to_world"])
        gt_centre = T_gt[:3, 3]

        # Ground-truth longitudinal direction, used only for evaluation.
        gt_axis = T_gt[:2, 0]
        gt_axis /= np.linalg.norm(gt_axis)

        axis_error = np.degrees(
            np.arccos(np.clip(abs(np.dot(axis, gt_axis)), -1, 1))
        )

        print(f"\nSample {sid}")
        print("Target points:", len(points))
        print("End-normal candidates:", n_candidates)
        print("Estimated axis:", np.round(axis, 4))
        print(f"Axis error: {axis_error:.3f} deg")
        print("Candidate planes:", len(planes))

        for i, plane in enumerate(planes):
            normal_xy = plane["normal"][:2]
            normal_xy /= np.linalg.norm(normal_xy)

            # Candidate centre: move inward by half the CAD length.
            estimated_centre = (
                plane["centroid"][:2] - 0.5 * LENGTH * normal_xy
            )

            translation_error = 1000 * np.linalg.norm(
                estimated_centre - gt_centre[:2]
            )

            print(
                f"  Plane {i}: N={plane['count']}, "
                f"dimensions={np.round(plane['dimensions_mm'], 1)} mm, "
                f"median residual={plane['median_residual_mm']:.2f} mm, "
                f"P95 residual={plane['p95_residual_mm']:.2f} mm"
            )
            print(
                f"    Estimated centre XY: "
                f"{np.round(estimated_centre, 4)}"
            )
            print(
                f"    Ground-truth centre XY: "
                f"{np.round(gt_centre[:2], 4)}"
            )
            print(
                f"    Translation error: {translation_error:.2f} mm"
            )


if __name__ == "__main__":
    main()