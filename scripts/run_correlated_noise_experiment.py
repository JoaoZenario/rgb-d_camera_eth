import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import (
    apply_depth_noise,
    DEFAULT_DISPARITY_SIGMA_PX,
    DEFAULT_DISPARITY_QUANTIZATION_PX,
    DEFAULT_CORRELATION_SIGMA_PX,
)
from src.experiments.obb_pose import (
    estimate_obb_pose,
    evaluate_pose,
)


DATA_DIR = Path("outputs/randomized_pose_dataset")
OUTPUT_DIR = Path("results/correlated_noise_obb")

BASE_SEED = 20261106
REPETITIONS = 5
TARGET_CATEGORY = 4

CROSS_AXIS_THRESHOLD_MM = 5.0
TRANSLATION_THRESHOLD_MM = 5.0
AXIS_THRESHOLD_DEG = 5.0

CORRELATION_SIGMA_PX = DEFAULT_CORRELATION_SIGMA_PX

CONDITIONS = ("independent", "correlated")

CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p95": float(np.percentile(values, 95)),
    }


def bootstrap_ci(values, seed, repetitions=10000):
    """Percentile bootstrap CI over independent pose-level observations."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)

    for i in range(repetitions):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)

    return {
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


def run_estimation(depth, segmentation, K, T_cw, T_wo):
    estimate = estimate_obb_pose(
        depth=depth,
        segmentation=segmentation,
        K=K,
        camera_to_world=T_cw,
        target_category=TARGET_CATEGORY,
    )

    return evaluate_pose(estimate, T_wo)


def paired_metric_comparison(pose_results, metric, seed):
    """Compare pose-level condition means on poses with both estimates."""
    deltas = []

    for result in pose_results:
        a = result["independent"].get(metric)
        b = result["correlated"].get(metric)

        if a is not None and b is not None:
            deltas.append(b - a)

    return {
        "paired_poses": len(deltas),
        "correlated_minus_independent": summarize(deltas),
        "bootstrap_95_ci": bootstrap_ci(deltas, seed=seed),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "pose_labels.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = sorted(
        metadata["samples"],
        key=lambda sample: int(sample["sample_id"]),
    )

    if not samples:
        raise RuntimeError("No samples found in pose_labels.json.")

    records = []
    per_pose = []

    print("Spatially correlated disparity-noise experiment")
    print("================================================")
    print("Samples:", len(samples))
    print("Repetitions per condition:", REPETITIONS)
    print("Conditions:", CONDITIONS)
    print("Disparity sigma:", DEFAULT_DISPARITY_SIGMA_PX, "px")
    print("Quantisation:", DEFAULT_DISPARITY_QUANTIZATION_PX, "px")
    print("Correlation sigma:", CORRELATION_SIGMA_PX, "px")
    print("Range dropout: disabled")
    print("Boundary dropout: disabled")
    print("Expected trials:", len(samples) * len(CONDITIONS) * REPETITIONS)

    for sample_index, sample in enumerate(samples):
        sid = int(sample["sample_id"])

        T_cw = np.asarray(
            sample["camera_to_world_blender"],
            dtype=float,
        )
        T_wo = np.asarray(
            sample["target_to_world"],
            dtype=float,
        )

        h5_path = DATA_DIR / f"{sid}.hdf5"

        if not h5_path.exists():
            raise FileNotFoundError(f"Missing sample: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            depth_ideal = np.squeeze(f["depth"][:]).astype(np.float32)
            segmentation = np.squeeze(
                f["category_id_segmaps"][:]
            )

        target_mask = segmentation == TARGET_CATEGORY

        original_target_count = int(np.count_nonzero(
            target_mask
            & np.isfinite(depth_ideal)
            & (depth_ideal > 0)
            & (depth_ideal < 1000)
        ))

        pose_record = {
            "sample_id": sid,
            "independent": {},
            "correlated": {},
        }

        # Same seed for both modes at each pose/repetition.
        # This gives the comparison a common random-number design.
        for repetition in range(REPETITIONS):
            seed = BASE_SEED + sample_index * 1009 + repetition

            for condition in CONDITIONS:
                depth_test, invalid_mask, edge_mask = apply_depth_noise(
                    depth_ideal,
                    segmentation,
                    seed=seed,
                    gaussian_noise=True,
                    range_missing=False,
                    boundary_failures=False,
                    disparity_noise_mode=condition,
                    correlation_sigma_px=CORRELATION_SIGMA_PX,
                )

                valid_target = (
                    target_mask
                    & np.isfinite(depth_test)
                    & (depth_test > 0)
                    & (depth_test < 1000)
                )

                valid_count = int(np.count_nonzero(valid_target))

                invalid_rate = (
                    1.0 - valid_count / original_target_count
                    if original_target_count > 0
                    else 1.0
                )

                record = {
                    "sample_id": sid,
                    "condition": condition,
                    "repetition": repetition,
                    "seed": seed,
                    "valid_target_points": valid_count,
                    "invalid_target_rate": invalid_rate,
                    "estimator_valid": False,
                    "translation_error_mm": "",
                    "along_axis_error_mm": "",
                    "cross_axis_error_mm": "",
                    "long_axis_error_deg": "",
                    "success_cross_axis_5mm_axis_5deg": False,
                    "success_translation_5mm_axis_5deg": False,
                    "error_message": "",
                }

                try:
                    errors = run_estimation(
                        depth_test,
                        segmentation,
                        K,
                        T_cw,
                        T_wo,
                    )

                    translation = float(errors["translation_error_mm"])
                    along = float(errors["along_axis_error_mm"])
                    cross = float(errors["cross_axis_error_mm"])
                    axis = float(errors["long_axis_error_deg"])

                    cross_success = (
                        cross <= CROSS_AXIS_THRESHOLD_MM
                        and axis <= AXIS_THRESHOLD_DEG
                    )

                    translation_success = (
                        translation <= TRANSLATION_THRESHOLD_MM
                        and axis <= AXIS_THRESHOLD_DEG
                    )

                    record.update({
                        "estimator_valid": True,
                        "translation_error_mm": translation,
                        "along_axis_error_mm": along,
                        "cross_axis_error_mm": cross,
                        "long_axis_error_deg": axis,
                        "success_cross_axis_5mm_axis_5deg": cross_success,
                        "success_translation_5mm_axis_5deg": translation_success,
                    })

                    pose_record[condition].setdefault(
                        "valid_trials", 0
                    )
                    pose_record[condition]["valid_trials"] += 1

                    for metric, value in (
                        ("translation", translation),
                        ("along", along),
                        ("cross", cross),
                        ("axis", axis),
                    ):
                        pose_record[condition].setdefault(
                            metric, []
                        )
                        pose_record[condition][metric].append(value)

                except Exception as exc:
                    record["error_message"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                # Failed estimates count as unsuccessful.
                pose_record[condition].setdefault(
                    "cross_success", []
                ).append(
                    bool(record["success_cross_axis_5mm_axis_5deg"])
                )
                pose_record[condition].setdefault(
                    "translation_success", []
                ).append(
                    bool(record["success_translation_5mm_axis_5deg"])
                )

                records.append(record)

        # Convert each pose's trial measurements to pose-level means.
        # Success rates include failed trials as unsuccessful.
        for condition in CONDITIONS:
            condition_data = pose_record[condition]

            for metric in ("translation", "along", "cross", "axis"):
                values = condition_data.get(metric, [])
                condition_data[metric] = (
                    float(np.mean(values)) if values else None
                )

            condition_data["coverage"] = (
                condition_data.get("valid_trials", 0) / REPETITIONS
            )
            condition_data["cross_success_rate"] = float(
                np.mean(condition_data["cross_success"])
            )
            condition_data["translation_success_rate"] = float(
                np.mean(condition_data["translation_success"])
            )

            del condition_data["cross_success"]
            del condition_data["translation_success"]

        per_pose.append(pose_record)

        if (sample_index + 1) % 10 == 0:
            print(
                f"Processed {sample_index + 1}/{len(samples)} poses "
                f"({len(records)} trials)"
            )

    # Save trial-level results.
    trial_path = OUTPUT_DIR / "trial_results.csv"

    with open(trial_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    # Save pose-level aggregates.
    pose_path = OUTPUT_DIR / "per_pose_results.csv"
    pose_rows = []

    for pose in per_pose:
        row = {"sample_id": pose["sample_id"]}

        for condition in CONDITIONS:
            for key, value in pose[condition].items():
                row[f"{condition}_{key}"] = value

        pose_rows.append(row)

    with open(pose_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(pose_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(pose_rows)

    # Trial-level descriptive statistics.
    summary = {
        "experiment": "independent_vs_spatially_correlated_disparity",
        "num_poses": len(samples),
        "repetitions_per_condition": REPETITIONS,
        "base_seed": BASE_SEED,
        "disparity_sigma_px": DEFAULT_DISPARITY_SIGMA_PX,
        "disparity_quantization_px": DEFAULT_DISPARITY_QUANTIZATION_PX,
        "correlation_sigma_px": CORRELATION_SIGMA_PX,
        "range_missing": False,
        "boundary_failures": False,
        "success_thresholds": {
            "cross_axis_mm": CROSS_AXIS_THRESHOLD_MM,
            "translation_mm": TRANSLATION_THRESHOLD_MM,
            "long_axis_deg": AXIS_THRESHOLD_DEG,
        },
        "conditions": {},
        "paired_comparisons": {},
    }

    for condition in CONDITIONS:
        subset = [
            row for row in records
            if row["condition"] == condition
        ]
        valid = [
            row for row in subset
            if row["estimator_valid"]
        ]

        summary["conditions"][condition] = {
            "trials": len(subset),
            "valid_estimates": len(valid),
            "coverage": len(valid) / len(subset),
            "translation_error_mm": summarize([
                row["translation_error_mm"] for row in valid
            ]),
            "along_axis_error_mm": summarize([
                row["along_axis_error_mm"] for row in valid
            ]),
            "cross_axis_error_mm": summarize([
                row["cross_axis_error_mm"] for row in valid
            ]),
            "long_axis_error_deg": summarize([
                row["long_axis_error_deg"] for row in valid
            ]),
            "cross_axis_geometric_success_rate": float(np.mean([
                row["success_cross_axis_5mm_axis_5deg"]
                for row in subset
            ])),
            "translation_geometric_success_rate": float(np.mean([
                row["success_translation_5mm_axis_5deg"]
                for row in subset
            ])),
        }

    # Paired comparisons use one aggregated observation per pose.
    for metric, seed_offset in (
        ("translation", 1),
        ("along", 2),
        ("cross", 3),
        ("axis", 4),
        ("coverage", 5),
        ("cross_success_rate", 6),
        ("translation_success_rate", 7),
    ):
        summary["paired_comparisons"][metric] = (
            paired_metric_comparison(
                per_pose,
                metric,
                seed=BASE_SEED + seed_offset,
            )
        )

    summary_path = OUTPUT_DIR / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCONDITION SUMMARY")
    print("=================")

    for condition, result in summary["conditions"].items():
        print(f"\n{condition}")
        print(
            "  Coverage:",
            f"{100 * result['coverage']:.2f}%",
        )
        print(
            "  Translation median / P95:",
            result["translation_error_mm"]["median"],
            "/",
            result["translation_error_mm"]["p95"],
            "mm",
        )
        print(
            "  Cross-axis median / P95:",
            result["cross_axis_error_mm"]["median"],
            "/",
            result["cross_axis_error_mm"]["p95"],
            "mm",
        )
        print(
            "  Along-axis median:",
            result["along_axis_error_mm"]["median"],
            "mm",
        )
        print(
            "  Axis median / P95:",
            result["long_axis_error_deg"]["median"],
            "/",
            result["long_axis_error_deg"]["p95"],
            "deg",
        )
        print(
            "  Cross-axis geometric success:",
            f"{100 * result['cross_axis_geometric_success_rate']:.2f}%",
        )
        print(
            "  Translation geometric success:",
            f"{100 * result['translation_geometric_success_rate']:.2f}%",
        )

    print("\nPAIRED EFFECTS: CORRELATED MINUS INDEPENDENT")
    print("===========================================")

    for metric, result in summary["paired_comparisons"].items():
        delta = result["correlated_minus_independent"]
        ci = result["bootstrap_95_ci"]

        print(f"\n{metric}")
        print("  Paired poses:", result["paired_poses"])
        print("  Mean change:", delta["mean"])
        print("  Bootstrap 95% CI:", ci)

    print("\nSaved:")
    print(trial_path)
    print(pose_path)
    print(summary_path)


if __name__ == "__main__":
    main()