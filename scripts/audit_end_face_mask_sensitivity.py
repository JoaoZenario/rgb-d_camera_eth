import csv
import json
import numpy as np

from scripts.test_cad_end_estimator import DATA, load_sample

HALF_LENGTH = 0.65
HALF_WIDTH = 0.04
HALF_HEIGHT = 0.04
FACE_TOLERANCE = 0.005

EDGE_MARGINS = [0.000, 0.002, 0.005, 0.010]


def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)

    results = []

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

        face_plane = (
            (np.abs(local[:, 0] - HALF_LENGTH) < FACE_TOLERANCE)
            |
            (np.abs(local[:, 0] + HALF_LENGTH) < FACE_TOLERANCE)
        )

        row = {
            "sample_id": sid,
            "target_points": len(points),
        }

        for margin in EDGE_MARGINS:
            interior = (
                (np.abs(local[:, 1]) < HALF_WIDTH - margin)
                & (np.abs(local[:, 2]) < HALF_HEIGHT - margin)
            )

            count = int((face_plane & interior).sum())
            key = f"margin_{int(margin * 1000)}mm"
            row[key] = count

        results.append(row)

    output = DATA.parent.parent / "results" / "end_face_mask_sensitivity"
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("Saved:", output / "results.csv")

    print("\nNumber of samples with at least 20 end-face points:")
    for margin in EDGE_MARGINS:
        key = f"margin_{int(margin * 1000)}mm"
        count = sum(r[key] >= 20 for r in results)
        print(f"  {int(margin * 1000):2d} mm margin: {count}/200")

    print("\nFailed samples with the largest increase when relaxing the mask:")
    failed_ids = {
        4, 12, 17, 22, 25, 26, 31, 42, 44, 45,
        47, 50, 55, 56, 62, 63, 64, 65, 66, 68,
        69, 70, 72, 79, 80, 82, 88, 94, 100, 106,
        108, 109, 110, 113, 118, 119, 120, 122,
        125, 128, 131, 135, 137, 145, 151, 152,
        154, 155, 157, 158, 160, 161, 167, 168,
        175, 176, 181, 182, 185, 186, 190, 191, 194,
    }

    failed = [r for r in results if r["sample_id"] in failed_ids]
    failed.sort(
        key=lambda r: r["margin_0mm"] - r["margin_5mm"],
        reverse=True,
    )

    for r in failed[:20]:
        print(
            f"  ID {r['sample_id']:3d}: "
            f"0mm={r['margin_0mm']:4d}, "
            f"2mm={r['margin_2mm']:4d}, "
            f"5mm={r['margin_5mm']:4d}, "
            f"10mm={r['margin_10mm']:4d}"
        )


if __name__ == "__main__":
    main()