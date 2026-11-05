import csv
import numpy as np
import json

from pathlib import Path
from scripts.test_cad_end_estimator import (
    DATA,
    load_sample,
    analyse,
)
from scripts.run_cad_end_experiment import select_end_plane

RADII = [0.025, 0.035, 0.050]
ANGLES = [20, 30, 40]

OUT = Path("results/normal_estimation_sweep")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)

    rows = []

    for radius in RADII:
        for angle in ANGLES:

            print(
                f"\nRadius={radius*1000:.0f} mm, "
                f"angle={angle} degrees"
            )

            for sample in metadata["samples"]:
                sid = int(sample["sample_id"])

                points, camera = load_sample(sample, K)

                axis, n_candidates, planes = analyse(
                    points,
                    camera,
                    normal_radius=radius,
                    angle_deg=angle,
                    min_candidates=20,
                )

                selected = select_end_plane(planes)

                rows.append({
                    "sample_id": sid,
                    "radius_mm": radius * 1000,
                    "angle_deg": angle,
                    "target_points": len(points),
                    "candidate_count": n_candidates,
                    "plane_count": len(planes),
                    "plausible_plane": selected is not None,
                    "selected_plane_points": (
                        selected["count"]
                        if selected is not None else 0
                    ),
                    "p95_residual_mm": (
                        selected["p95_residual_mm"]
                        if selected is not None else np.nan
                    ),
                })

    path = OUT / "per_sample.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nCONFIGURATION SUMMARY")

    print(
        f"{'Radius':>8} {'Angle':>8} {'Candidates':>12} "
        f"{'Plausible planes':>17} {'Median candidates':>19}"
    )

    for radius in RADII:
        for angle in ANGLES:
            subset = [
                r for r in rows
                if r["radius_mm"] == radius * 1000
                and r["angle_deg"] == angle
            ]

            enough_candidates = sum(
                r["candidate_count"] >= 20
                for r in subset
            )

            plausible = sum(
                r["plausible_plane"]
                for r in subset
            )

            median_candidates = np.median([
                r["candidate_count"] for r in subset
            ])

            print(
                f"{radius*1000:7.0f}mm "
                f"{angle:7.0f}° "
                f"{enough_candidates:9d}/200 "
                f"{plausible:12d}/200 "
                f"{median_candidates:17.0f}"
            )

    print("\nSaved:", path)


if __name__ == "__main__":
    main()