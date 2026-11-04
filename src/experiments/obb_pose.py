import numpy as np


CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def estimate_obb_pose(
    depth,
    segmentation,
    K,
    camera_to_world,
    target_category=4,
    min_points=100,
):
    """
    Estimate target centre and undirected long-axis direction
    from a segmented axial depth map.

    Returns centre, long axis, point count, extents and eigenvalues.
    """

    valid = (
        (segmentation == target_category)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)
    z = depth[v, u].astype(np.float64)

    if len(z) < min_points:
        raise ValueError(
            f"Too few valid target points: {len(z)}"
        )

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    points_camera_blender = (
        CV_TO_BLENDER @ points_cv.T
    ).T

    points_world = (
        camera_to_world @ points_camera_blender.T
    ).T[:, :3]

    mean = points_world.mean(axis=0)
    centered = points_world - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    R_pca = eigenvectors[:, order]

    if np.linalg.det(R_pca) < 0:
        R_pca[:, 2] *= -1

    local = centered @ R_pca

    local_min = local.min(axis=0)
    local_max = local.max(axis=0)

    local_mid = (local_min + local_max) / 2.0
    centre = mean + R_pca @ local_mid

    return {
        "centre_world": centre,
        "long_axis_world": R_pca[:, 0],
        "point_count": int(len(z)),
        "extents_m": local_max - local_min,
        "eigenvalues": eigenvalues,
    }


def evaluate_pose(estimate, T_target_gt):
    """Evaluate translation and undirected long-axis errors."""

    R_gt = T_target_gt[:3, :3]
    t_gt = T_target_gt[:3, 3]

    translation_error_mm = (
        np.linalg.norm(estimate["centre_world"] - t_gt) * 1000.0
    )

    axis_cosine = np.clip(
        abs(np.dot(
            estimate["long_axis_world"],
            R_gt[:, 0]
        )),
        0.0,
        1.0,
    )

    axis_error_deg = np.degrees(np.arccos(axis_cosine))

    return {
        "translation_error_mm": float(translation_error_mm),
        "long_axis_error_deg": float(axis_error_deg),
    }