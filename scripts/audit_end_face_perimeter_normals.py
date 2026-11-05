import csv
import json
import numpy as np
import open3d as o3d

from scripts.test_cad_end_estimator import DATA, load_sample

HALF_LENGTH = 0.65
HALF_WIDTH = 0.04
HALF_HEIGHT = 0.04
FACE_TOLERANCE = 0.005

NORMAL_RADIUS = 0.025
ANGLE_DEG = 20

# Distance from the nearest edge of the end face.
REGIONS = [
    ("0-2mm", 0.000, 0.002),
    ("2-5mm", 0.002, 0.005),
    (">5mm", 0.005, float("inf")),
]

FAILED_IDS = {
    4, 12, 17, 22, 25, 26, 31, 42, 44, 45,
    47, 50, 55, 56, 62, 63, 64, 65, 66, 68,
    69, 70, 72, 79, 80, 82, 88, 94, 100, 106,
    108, 109, 110, 113, 118, 119, 120, 122,
    125, 128, 131, 135, 137, 145, 151, 152,
    154, 155, 157, 158, 160, 161, 167, 168,
    175, 176, 181, 182, 185, 186, 190, 191, 194,
}


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    threshold = np.cos(np.deg2rad(ANGLE_DEG))

    output = (
        DATA.parent.parent
        / "results"
        / "end_face_perimeter_normals"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    for sample in metadata["samples"]:
        sid = int(sample["sample_id"])
        points, _ = load_sample(sample, K)

        T = np.asarray(sample["target_to_world"])

        local = (
            np.linalg.inv(T)
            @ np.column_stack([
                points, np.ones(len(points))
            ]).T
        ).T[:, :3]

        # Estimate normals on the complete target point cloud.
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

        gt_axis = T[:2, 0].copy()
        gt_axis /= np.linalg.norm(gt_axis)

        alignment = np.zeros(len(points))
        alignment[valid] = np.abs(
            (
                horizontal[valid]
                / horizontal_norm[valid, None]
            ) @ gt_axis
        )

        # Points close to either end plane.
        on_end_plane = (
            (
                np.abs(local[:, 0] - HALF_LENGTH)
                < FACE_TOLERANCE
            )
            |
            (
                np.abs(local[:, 0] + HALF_LENGTH)
                < FACE_TOLERANCE
            )
        )

        # Signed distance to the nearest nominal face boundary.
        edge_distance = np.minimum(
            HALF_WIDTH - np.abs(local[:, 1]),
            HALF_HEIGHT - np.abs(local[:, 2]),
        )

        inside_face = (
            (edge_distance >= 0)
            & on_end_plane
        )

        for region_name, lower, upper in REGIONS:
            mask = (
                inside_face
                & (edge_distance >= lower)
                & (edge_distance < upper)
            )

            n = int(mask.sum())
            valid_mask = mask & valid
            n_valid = int(valid_mask.sum())

            if n_valid:
                values = alignment[valid_mask]
                median_alignment = float(np.median(values))
                aligned_pct = float(
                    100 * np.mean(values > threshold)
                )
            else:
                median_alignment = float("nan")
                aligned_pct = float("nan")

            rows.append({
                "sample_id": sid,
                "condition": (
                    "failed" if sid in FAILED_IDS else "control"
                ),
                "region": region_name,
                "points": n,
                "valid_normals": n_valid,
                "median_alignment": median_alignment,
                "within_20deg_pct": aligned_pct,
            })

    with open(output / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output / "results.csv")

    print("\nSelected failed samples:")
    print(
        f"{'ID':>4} {'Region':>7} {'Points':>7} "
        f"{'Valid':>7} {'Median':>8} {'Within20':>10}"
    )

    selected = {17, 56, 65, 79, 108, 113, 120, 181, 190, 194}

    for row in rows:
        if row["sample_id"] not in selected:
            continue

        print(
            f"{row['sample_id']:4d} "
            f"{row['region']:>7} "
            f"{row['points']:7d} "
            f"{row['valid_normals']:7d} "
            f"{row['median_alignment']:8.3f} "
            f"{row['within_20deg_pct']:9.1f}%"
        )


if __name__ == "__main__":
    main()