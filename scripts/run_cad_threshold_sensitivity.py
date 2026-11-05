import csv
import json
import numpy as np

from pathlib import Path
from scripts.run_cad_end_experiment import (
    CONDITIONS,
    NOISY_REPETITIONS,
    DATA,
    evaluate_sample,
)

OUT = Path("results/cad_threshold_sensitivity")

THRESHOLDS = [5, 10, 15, 20]

# Two separate experiments:
# A: vary total candidates; keep cluster size at 20.
# B: vary cluster size; keep total candidates at 20.
SWEEPS = {
    "total_candidate_threshold": [
        (value, 20) for value in THRESHOLDS
    ],
    "cluster_size_threshold": [
        (20, value) for value in THRESHOLDS
    ],
}

# Ideal and full conditions provide a useful comparison
# without repeating all intermediate ablations.
SELECTED_CONDITIONS = ["ideal", "full"]


def summarise(rows):
    valid = [r for r in rows if r["success"]]

    translations = np.array([
        r["translation_error_mm"] for r in valid
    ])

    angles = np.array([
        r["axis_error_deg"] for r in valid
    ])

    successes_5 = sum(
        r["success"]
        and r["translation_error_mm"] <= 5
        and r["axis_error_deg"] <= 5
        for r in rows
    )

    successes_10 = sum(
        r["success"]
        and r["translation_error_mm"] <= 10
        and r["axis_error_deg"] <= 5
        for r in rows
    )

    def pct(values, q):
        return (
            float(np.percentile(values, q))
            if len(values) else None
        )

    n = len(rows)

    return {
        "trials": n,
        "returned_estimates": len(valid),
        "coverage": len(valid) / n if n else None,
        "translation_median_mm": pct(translations, 50),
        "translation_p95_mm": pct(translations, 95),
        "axis_median_deg": pct(angles, 50),
        "axis_p95_deg": pct(angles, 95),
        "success_5mm_5deg": successes_5,
        "success_5mm_5deg_fraction": successes_5 / n if n else None,
        "success_10mm_5deg": successes_10,
        "success_10mm_5deg_fraction": successes_10 / n if n else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = metadata["samples"]

    all_rows = []
    summary = {}

    total_trials = 0

    for sweep_name, configurations in SWEEPS.items():
        summary[sweep_name] = {}

        for min_total, min_cluster in configurations:
            config_name = (
                f"total_{min_total}_cluster_{min_cluster}"
            )

            rows = []

            for condition in SELECTED_CONDITIONS:
                repetitions = (
                    1 if condition == "ideal"
                    else NOISY_REPETITIONS
                )

                for repetition in range(repetitions):
                    for sample in samples:
                        result = evaluate_sample(
                            sample,
                            K,
                            condition,
                            repetition,
                            min_total_candidates=min_total,
                            min_cluster_points=min_cluster,
                        )

                        result["sweep"] = sweep_name
                        result["configuration"] = config_name

                        rows.append(result)
                        all_rows.append(result)
                        total_trials += 1

                        if total_trials % 100 == 0:
                            print(
                                f"Completed {total_trials} trials; "
                                f"{sweep_name}, {config_name}, "
                                f"{condition}, sample "
                                f"{result['sample_id']}"
                            )

            summary[sweep_name][config_name] = {}

            for condition in SELECTED_CONDITIONS:
                subset = [
                    r for r in rows
                    if r["condition"] == condition
                ]

                summary[sweep_name][config_name][condition] = (
                    summarise(subset)
                )

                stats = summary[sweep_name][config_name][condition]

                print(
                    f"\n{sweep_name} | {config_name} | {condition}"
                )
                print(
                    f"Coverage: {stats['returned_estimates']}/"
                    f"{stats['trials']} "
                    f"({100 * stats['coverage']:.1f}%)"
                )
                print(
                    "Translation median / P95:",
                    stats["translation_median_mm"],
                    "/",
                    stats["translation_p95_mm"],
                    "mm",
                )
                print(
                    "5 mm / 5 deg overall success:",
                    f"{100 * stats['success_5mm_5deg_fraction']:.1f}%",
                )

    csv_path = OUT / "per_trial_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = OUT / "summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()