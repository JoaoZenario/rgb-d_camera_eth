import csv
import json
import numpy as np

from collections import defaultdict
from pathlib import Path

INPUT = Path("outputs/randomized_pose_ablation/trial_results.csv")
OUTPUT = Path("outputs/randomized_pose_ablation")

TRANSLATION_THRESHOLDS_MM = [5, 10, 15, 20, 25]
AXIS_THRESHOLD_DEG = 5.0
BOOTSTRAP_REPETITIONS = 10000
SEED = 20261106


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    estimates = np.empty(BOOTSTRAP_REPETITIONS)

    for i in range(BOOTSTRAP_REPETITIONS):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[i] = np.mean(sample)

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def main():
    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # Group measurements by condition and sample.
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row["estimator_valid"].lower() != "true":
            continue

        grouped[row["condition"]][int(row["sample_id"])].append({
            "translation": float(row["translation_error_mm"]),
            "axis": float(row["long_axis_error_deg"]),
        })

    conditions = [
        "ideal",
        "disparity_only",
        "dropout_only",
        "full_corruption",
    ]

    rng = np.random.default_rng(SEED)
    output_rows = []

    print("GRASP TOLERANCE SENSITIVITY")
    print("==========================")
    print("Axis threshold:", AXIS_THRESHOLD_DEG, "degrees")

    for threshold in TRANSLATION_THRESHOLDS_MM:
        print(f"\nTranslation threshold: {threshold} mm")

        for condition in conditions:
            per_pose_success = []

            for sid, trials in grouped[condition].items():
                successes = [
                    float(
                        x["translation"] <= threshold
                        and x["axis"] <= AXIS_THRESHOLD_DEG
                    )
                    for x in trials
                ]

                # Average the five corruption repetitions within each pose.
                per_pose_success.append(np.mean(successes))

            per_pose_success = np.asarray(per_pose_success)
            rate = float(np.mean(per_pose_success))
            lo, hi = bootstrap_ci(per_pose_success, rng)

            output_rows.append({
                "translation_threshold_mm": threshold,
                "axis_threshold_deg": AXIS_THRESHOLD_DEG,
                "condition": condition,
                "num_poses": len(per_pose_success),
                "success_rate": rate,
                "ci_lower_95": lo,
                "ci_upper_95": hi,
            })

            print(
                f"  {condition:18s}: {100*rate:6.2f}% "
                f"(95% CI {100*lo:.2f}–{100*hi:.2f}%)"
            )

    # Save machine-readable results.
    csv_path = OUTPUT / "grasp_tolerance_sensitivity.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSaved:", csv_path)

    # Plot the sensitivity curve.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.5))

        for condition in conditions:
            subset = [
                r for r in output_rows
                if r["condition"] == condition
            ]

            x = [r["translation_threshold_mm"] for r in subset]
            y = [100 * r["success_rate"] for r in subset]
            lower = [
                100 * (r["success_rate"] - r["ci_lower_95"])
                for r in subset
            ]
            upper = [
                100 * (r["ci_upper_95"] - r["success_rate"])
                for r in subset
            ]

            ax.plot(x, y, marker="o", label=condition)
            ax.fill_between(
                x,
                np.asarray(y) - np.asarray(lower),
                np.asarray(y) + np.asarray(upper),
                alpha=0.12,
            )

        ax.set_xlabel("Translation tolerance (mm)")
        ax.set_ylabel("Grasp success rate (%)")
        ax.set_title("Grasp success versus translation tolerance")
        ax.set_xticks(TRANSLATION_THRESHOLDS_MM)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        figure_path = OUTPUT / "grasp_tolerance_sensitivity.png"
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)

        print("Saved:", figure_path)

    except ImportError:
        print("Matplotlib unavailable; CSV results were saved.")


if __name__ == "__main__":
    main()