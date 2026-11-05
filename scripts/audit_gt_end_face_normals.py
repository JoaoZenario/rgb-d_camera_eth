import json
import numpy as np
import open3d as o3d

from scripts.test_cad_end_estimator import DATA, load_sample

HALF_LENGTH = 0.65
FACE_TOLERANCE = 0.005
HALF_WIDTH = 0.04
HALF_HEIGHT = 0.04
EDGE_MARGIN = 0.005

FAILED_IDS = [
    4, 12, 17, 22, 25, 26, 31, 42, 44, 45,
    47, 50, 55, 56, 62, 63, 64, 65, 66, 68,
    69, 70, 72, 79, 80, 82, 88, 94, 100, 106,
    109, 110, 118, 119, 120, 122, 125, 128,
    131, 135, 137, 145, 151, 152, 154, 155,
    157, 158, 160, 161, 167, 168, 175, 176,
    182, 185, 186, 190, 191, 194,
]

CONTROL_IDS = [0, 17, 189, 198]


def main():
    print("Ground-truth end-face membership and axis are used for diagnosis only.")
    print("This is not an estimator result.")

    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = {
        int(s["sample_id"]): s
        for s in metadata["samples"]
    }

    for radius in [0.025, 0.050]:
        print(f"\n{'=' * 65}")
        print(f"Normal radius: {radius * 1000:.0f} mm")
        print(
            f"{'ID':>4} {'Face':>5} {'Points':>7} {'Valid':>7} "
            f"{'Median':>8} {'>20°':>8} {'>30°':>8} {'>40°':>8}"
        )

        for sid in sorted(set(FAILED_IDS) | set(CONTROL_IDS)):
            points, _ = load_sample(samples[sid], K)

            T = np.asarray(samples[sid]["target_to_world"])
            local = (
                np.linalg.inv(T)
                @ np.column_stack([points, np.ones(len(points))]).T
            ).T[:, :3]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius,
                    max_nn=50,
                )
            )

            normals = np.asarray(cloud.normals)
            horizontal = normals[:, :2]
            horizontal_norm = np.linalg.norm(horizontal, axis=1)

            # Ground-truth longitudinal axis, used only for this audit.
            gt_axis = T[:2, 0]
            gt_axis /= np.linalg.norm(gt_axis)

            alignment = np.zeros(len(points))
            valid = horizontal_norm > 0.5
            alignment[valid] = np.abs(
                (horizontal[valid] / horizontal_norm[valid, None])
                @ gt_axis
            )

            interior_mask = (
                (np.abs(local[:, 1]) < HALF_WIDTH - EDGE_MARGIN)
                & (np.abs(local[:, 2]) < HALF_HEIGHT - EDGE_MARGIN)
            )

            for face_name, face_mask in [
                (
                    "-X",
                    (np.abs(local[:, 0] + HALF_LENGTH) < FACE_TOLERANCE)
                    & interior_mask,
                ),
                (
                    "+X",
                    (np.abs(local[:, 0] - HALF_LENGTH) < FACE_TOLERANCE)
                    & interior_mask,
                ),
            ]:
                n = int(face_mask.sum())
                face_valid = face_mask & valid
                n_valid = int(face_valid.sum())

                if n_valid == 0:
                    continue

                values = alignment[face_valid]
                median = np.median(values)

                fractions = [
                    100 * np.mean(values < np.cos(np.deg2rad(a)))
                    for a in [20, 30, 40]
                ]

                print(
                    f"{sid:4d} {face_name:>5} {n:7d} {n_valid:7d} "
                    f"{median:8.3f} "
                    f"{fractions[0]:7.1f}% "
                    f"{fractions[1]:7.1f}% "
                    f"{fractions[2]:7.1f}%"
                )

if __name__ == "__main__":
    main()