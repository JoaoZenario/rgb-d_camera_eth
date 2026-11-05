import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scripts.test_cad_end_estimator import (
    DATA,
    load_sample,
    analyse,
)

OUTPUT = Path("report/figures/cad_end_estimator_examples.png")

SUCCESS_SAMPLE = 0
FAILURE_SAMPLE = 108

MIN_CANDIDATES = 20
MIN_CLUSTER_POINTS = 20
NORMAL_RADIUS = 0.025
ANGLE_DEG = 20
EXPECTED_FACE_MM = 80.0


def get_candidate_indices(points, axis):
    """Reproduce the estimator's end-normal candidate selection."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=50,
        )
    )

    normals = np.asarray(cloud.normals)
    horizontal = normals[:, :2].copy()
    norm = np.linalg.norm(horizontal, axis=1)

    valid = norm > 0.5
    horizontal[valid] /= norm[valid, None]

    alignment = np.abs(horizontal @ axis)

    return np.where(
        valid & (alignment > np.cos(np.deg2rad(ANGLE_DEG)))
    )[0]


def select_end_plane(planes):
    """Use the same geometric plane-selection criteria as the experiment."""
    candidates = []

    for plane in planes:
        dims = np.sort(plane["dimensions_mm"])
        thickness = dims[0]
        face_dims = dims[1:]

        if thickness > 12:
            continue

        if np.any(face_dims < 30) or np.any(face_dims > 110):
            continue

        dimension_error = np.sum(
            np.abs(face_dims - EXPECTED_FACE_MM)
        )

        score = (
            dimension_error
            + 2.0 * plane["p95_residual_mm"]
        )

        candidates.append((score, plane))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def add_plane(ax, plane):
    """Draw the fitted plane with a more visible patch and boundary."""
    centroid = plane["centroid"]
    normal = plane["normal"]

    # Construct two orthogonal directions in the plane.
    _, _, vt = np.linalg.svd(
        np.eye(3) - np.outer(normal, normal)
    )

    u_axis = vt[0]
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    # Square patch visualises the fitted plane.
    half_size = 0.055

    corners = np.array([
        centroid - half_size * u_axis - half_size * v_axis,
        centroid + half_size * u_axis - half_size * v_axis,
        centroid + half_size * u_axis + half_size * v_axis,
        centroid - half_size * u_axis + half_size * v_axis,
    ])

    polygon = Poly3DCollection(
        [corners],
        alpha=0.55,
        facecolor="royalblue",
        edgecolor="navy",
        linewidth=2.0,
        label="Fitted end-face plane",
    )

    ax.add_collection3d(polygon)

def set_fitted_axes(ax, points):
    """Fit the cloud while keeping readable, equally scaled axes."""
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)

    spans = upper - lower
    margins = np.maximum(spans * 0.12, 0.008)

    lower -= margins
    upper += margins

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])

    ax.set_box_aspect(upper - lower)

    # Fewer, consistently formatted tick labels.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
        axis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        pad=0,
    )

    ax.tick_params(
        axis="z",
        which="major",
        labelsize=8,
        pad=1,
    )

    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.zaxis.label.set_size(9)


def plot_sample(ax, sample, sample_id, successful):
    points, camera = load_sample(
        sample,
        K,
    )

    # Lower the initial candidate-count gate only to inspect sparse
    # candidates in the failure example. The operational threshold
    # is applied explicitly below.
    axis, _, planes = analyse(
        points,
        camera,
        normal_radius=NORMAL_RADIUS,
        angle_deg=ANGLE_DEG,
        min_candidates=1,
        min_cluster_points=MIN_CLUSTER_POINTS,
    )

    candidate_indices = get_candidate_indices(points, axis)
    n_candidates = len(candidate_indices)

    # Apply the actual experiment's minimum candidate threshold.
    accepted_by_candidate_gate = (
        n_candidates >= MIN_CANDIDATES
    )

    selected = (
        select_end_plane(planes)
        if accepted_by_candidate_gate
        else None
    )

    # Full target point cloud.
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1.5,
        c="0.72",
        alpha=0.32,
        linewidths=0,
        label="Target points",
    )

    # Points selected by the end-normal criterion.
    candidates = points[candidate_indices]

    if len(candidates):
        ax.scatter(
            candidates[:, 0],
            candidates[:, 1],
            candidates[:, 2],
            s=13,
            c="tab:orange",
            alpha=1.0,
            linewidths=0,
            label="End-normal candidates",
            depthshade=False,
        )

    # Estimated longitudinal axis.
    origin = np.median(points, axis=0)

    axis_start = origin.copy()
    axis_end = origin.copy()

    axis_start[:2] -= 0.35 * axis
    axis_end[:2] += 0.35 * axis

    ax.plot(
        [axis_start[0], axis_end[0]],
        [axis_start[1], axis_end[1]],
        [axis_start[2], axis_end[2]],
        linestyle="--",
        color="black",
        linewidth=2.0,
        label="Estimated longitudinal axis",
    )

    if successful and selected is not None:
        add_plane(ax, selected)

        centroid = selected["centroid"]

        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            marker="x",
            s=65,
            c="navy",
            linewidths=2.2,
            label="Fitted plane centre",
            depthshade=False,
        )

        subtitle = (
            f"{n_candidates} candidates | "
            f"{selected['count']} plane points | "
            f"P95: {selected['p95_residual_mm']:.2f} mm"
        )

    else:
        subtitle = (
            f"{n_candidates} candidates | "
            f"20 required | Rejected"
        )

    ax.set_xlabel("X (m)", labelpad=3)
    ax.set_ylabel("Y (m)", labelpad=3)
    ax.set_zlabel("Z (m)", labelpad=3)

    ax.view_init(elev=20, azim=-60)

    set_fitted_axes(ax, points)

    return n_candidates, selected, subtitle


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    global K
    K = np.asarray(metadata["intrinsics"], dtype=float)

    samples = {
        int(s["sample_id"]): s
        for s in metadata["samples"]
    }

    fig = plt.figure(figsize=(15, 7.0))

    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    left_count, left_plane, left_subtitle = plot_sample(
        ax_left,
        samples[SUCCESS_SAMPLE],
        SUCCESS_SAMPLE,
        successful=True,
    )

    right_count, right_plane, right_subtitle = plot_sample(
        ax_right,
        samples[FAILURE_SAMPLE],
        FAILURE_SAMPLE,
        successful=False,
    )

    # Separate panel headings and subtitles from the 3D axes.
    # This avoids the previous title/subtitle overlap.
    fig.text(
        0.27, 0.965,
        f"Successful estimate — sample {SUCCESS_SAMPLE}",
        ha="center", va="top",
        fontsize=12, fontweight="bold",
    )

    fig.text(
        0.27, 0.925,
        left_subtitle,
        ha="center", va="top",
        fontsize=9,
    )

    fig.text(
        0.73, 0.965,
        f"Rejected estimate — sample {FAILURE_SAMPLE}",
        ha="center", va="top",
        fontsize=12, fontweight="bold",
    )

    fig.text(
        0.73, 0.925,
        right_subtitle,
        ha="center", va="top",
        fontsize=9,
    )
    # Shared legend avoids repeating entries in both panels.
    handles, labels = ax_left.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.86,
        bottom=0.16,
        wspace=0.12,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved figure: {OUTPUT}")

    print(
        f"Sample {SUCCESS_SAMPLE}: "
        f"{left_count} candidates, "
        f"plane accepted={left_plane is not None}"
    )

    print(
        f"Sample {FAILURE_SAMPLE}: "
        f"{right_count} candidates, "
        f"operational threshold={MIN_CANDIDATES}, "
        f"plane accepted={right_plane is not None}"
    )

    plt.show()


if __name__ == "__main__":
    main()