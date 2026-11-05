import json
import csv
import numpy as np

from pathlib import Path
from scripts.test_cad_end_estimator import DATA, load_sample

HALF_LENGTH = 0.65
FACE_TOLERANCE = 0.005  # 5 mm
FAILURE_FILE = Path("results/normal_estimation_sweep/per_sample.csv")


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = {
        int(s["sample_id"]): s
        for s in metadata["samples"]
    }

    # Samples that failed all nine normal-estimation configurations.
    failed_ids = {
        4, 12, 17, 22, 25, 26, 31, 42, 44, 45, 47, 50,
        55, 56, 62, 63, 64, 65, 66, 68, 69, 70, 72,
        79, 80, 82, 88, 94, 100, 106, 109, 110, 118,
        119, 120, 122, 125, 128, 131, 135, 137, 145,
        151, 152, 154, 155, 157, 158, 160, 161, 167,
        168, 175, 176, 182, 185, 186, 190, 191, 194,
    }

    selected_ids = sorted(failed_ids | {0, 1, 19, 126, 189, 198})

    print(
        f"{'ID':>4} {'Points':>7} {'-X face':>9} "
        f"{'+X face':>9} {'Visible ends':>14}"
    )

    for sid in selected_ids:
        sample = samples[sid]
        points, camera = load_sample(sample, K)

        T = np.asarray(sample["target_to_world"])
        T_inv = np.linalg.inv(T)

        points_h = np.column_stack([
            points,
            np.ones(len(points))
        ])
        local = (T_inv @ points_h.T).T[:, :3]

        negative_end = (
            np.abs(local[:, 0] + HALF_LENGTH) < FACE_TOLERANCE
        )
        positive_end = (
            np.abs(local[:, 0] - HALF_LENGTH) < FACE_TOLERANCE
        )

        n_neg = int(negative_end.sum())
        n_pos = int(positive_end.sum())

        visible = []
        if n_neg >= 10:
            visible.append("-X")
        if n_pos >= 10:
            visible.append("+X")

        print(
            f"{sid:4d} {len(points):7d} {n_neg:9d} "
            f"{n_pos:9d} {','.join(visible) if visible else 'neither':>14}"
        )


if __name__ == "__main__":
    main()