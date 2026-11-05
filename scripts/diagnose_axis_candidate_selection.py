import csv
import json
import numpy as np
import open3d as o3d

from scripts.test_cad_end_estimator import DATA, load_sample

FAILED_IDS = [
    4, 12, 17, 22, 25, 26, 31, 42, 44, 45,
    47, 50, 55, 56, 62, 63, 64, 65, 66, 68,
    69, 70, 72, 79, 80, 82, 88, 94, 100, 106,
    108, 109, 110, 113, 118, 119, 120, 122,
    125, 128, 131, 135, 137, 145, 151, 152,
    154, 155, 157, 158, 160, 161, 167, 168,
    175, 176, 181, 182, 185, 186, 190, 191, 194,
]

CONTROL_IDS = [0, 1, 2, 19, 126, 189, 198]

NORMAL_RADIUS = 0.025
ANGLE_DEG = 20
HALF_LENGTH = 0.65
HALF_WIDTH = 0.04
HALF_HEIGHT = 0.04
FACE_TOLERANCE = 0.005
EDGE_MARGIN = 0.005


def estimate_axis(points):
    xy = points[:, :2]
    centred = xy - np.median(xy, axis=0)
    keep = np.ones(len(xy), dtype=bool)

    for _ in range(3):
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

    _, _, vt = np.linalg.svd(
        centred[keep], full_matrices=False
    )
    axis = vt[0]
    return axis / np.linalg.norm(axis)


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = {
        int(s["sample_id"]): s
        for s in metadata["samples"]
    }

    threshold = np.cos(np.deg2rad(ANGLE_DEG))

    results = []

    print("Axis-dependent candidate-selection diagnostic")
    print(f"Normal radius: {NORMAL_RADIUS * 1000:.0f} mm")
    print(f"Alignment threshold: {ANGLE_DEG} degrees")
    print("Ground truth is used only for offline diagnosis.\n")

    header = (
        f"{'ID':>4} {'Class':>7} {'AxisErr':>8} "
        f"{'EstCand':>8} {'GTCand':>8} "
        f"{'FacePts':>8} {'FaceEst':>8} {'FaceGT':>8}"
    )
    print(header)
    print("-" * len(header))

    for sid in sorted(set(FAILED_IDS) | set(CONTROL_IDS)):
        sample = samples[sid]
        points, _ = load_sample(sample, K)

        # Estimate the longitudinal axis.
        estimated_axis = estimate_axis(points)

        # Ground-truth axis, for diagnosis only.
        T_gt = np.asarray(sample["target_to_world"])
        gt_axis = T_gt[:2, 0].copy()
        gt_axis /= np.linalg.norm(gt_axis)

        axis_error = np.degrees(
            np.arccos(
                np.clip(abs(np.dot(estimated_axis, gt_axis)), -1, 1)
            )
        )

        # Estimate normals once, using the same cloud for both axes.
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS,
                max_nn=50,
            )
        )

        normals = np.asarray(cloud.normals)
        horizontal = normals[:, :2]
        horizontal_norm = np.linalg.norm(horizontal, axis=1)

        valid = horizontal_norm > 0.5
        horizontal_unit = np.zeros_like(horizontal)
        horizontal_unit[valid] = (
            horizontal[valid] / horizontal_norm[valid, None]
        )

        # Apply the estimator's exact candidate-selection rule.
        estimated_alignment = np.abs(
            horizontal_unit @ estimated_axis
        )
        gt_alignment = np.abs(
            horizontal_unit @ gt_axis
        )

        estimated_candidates = (
            valid & (estimated_alignment > threshold)
        )
        gt_candidates = valid & (gt_alignment > threshold)

        # Ground-truth end-face membership, for diagnosis only.
        local = (
            np.linalg.inv(T_gt)
            @ np.column_stack([
                points, np.ones(len(points))
            ]).T
        ).T[:, :3]

        interior = (
            (np.abs(local[:, 1]) < HALF_WIDTH - EDGE_MARGIN)
            & (np.abs(local[:, 2]) < HALF_HEIGHT - EDGE_MARGIN)
        )

        end_face = (
            (
                np.abs(local[:, 0] - HALF_LENGTH)
                < FACE_TOLERANCE
            )
            |
            (
                np.abs(local[:, 0] + HALF_LENGTH)
                < FACE_TOLERANCE
            )
        ) & interior

        face_points = int(end_face.sum())
        face_estimated = int(
            (end_face & estimated_candidates).sum()
        )
        face_gt = int(
            (end_face & gt_candidates).sum()
        )

        label = "FAIL" if sid in FAILED_IDS else "CONTROL"

        print(
            f"{sid:4d} {label:>7} {axis_error:8.2f} "
            f"{estimated_candidates.sum():8d} "
            f"{gt_candidates.sum():8d} "
            f"{face_points:8d} "
            f"{face_estimated:8d} "
            f"{face_gt:8d}"
        )

        results.append({
            "sample_id": sid,
            "class": label,
            "axis_error_deg": float(axis_error),
            "estimated_axis_candidates": int(
                estimated_candidates.sum()
            ),
            "gt_axis_candidates": int(gt_candidates.sum()),
            "visible_gt_end_face_points": face_points,
            "end_face_selected_estimated_axis": face_estimated,
            "end_face_selected_gt_axis": face_gt,
        })

    output = DATA.parent.parent / "results" / "axis_candidate_diagnostic"
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to:")
    print(output / "results.csv")


if __name__ == "__main__":
    main()