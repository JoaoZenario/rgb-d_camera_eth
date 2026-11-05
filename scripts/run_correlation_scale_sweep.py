import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import (
    apply_depth_noise,
    DEFAULT_DISPARITY_SIGMA_PX,
    DEFAULT_DISPARITY_QUANTIZATION_PX,
)
from src.experiments.obb_pose import (
    estimate_obb_pose,
    evaluate_pose,
)


DATA_DIR = Path("outputs/randomized_pose_dataset")
OUTPUT_DIR = Path("results/correlation_scale_sweep")

BASE_SEED = 20261106
REPETITIONS = 5
TARGET_CATEGORY = 4

CORRELATION_SCALES_PX = (2.0, 4.0, 8.0, 16.0)
CONDITIONS = (
    "independent",
    "correlated_2px",
    "correlated_4px",
    "correlated_8px",
    "correlated_16px",
)

CROSS_AXIS_THRESHOLD_MM = 5.0
TRANSLATION_THRESHOLD_MM = 5.0
AXIS_THRESHOLD_DEG = 5.0


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
    """Bootstrap confidence interval over independent pose-level values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)

    for i in range(repetitions):
        means[i] = np.mean(
            rng.choice(values, size=len(values), replace=True)
        )

    return {
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


def estimate_errors(depth, segmentation, K, T_cw, T_wo):
    estimate = estimate_obb_pose(
        depth=depth,
        segmentation=segmentation,
        K=K,
        camera_to_world=T_cw,
        target_category=TARGET_CATEGORY,
    )
    return evaluate_pose(estimate, T_wo)


def paired_comparison(per_pose, condition, metric, seed):
    deltas = []

    for pose in per_pose:
        reference = pose["independent"].get(metric)
        candidate = pose[condition].get(metric)

        if reference is not None and candidate is not None:
            deltas.append(candidate - reference)

    return {
        "paired_poses": len(deltas),
        "mean_delta_candidate_minus_independent": summarize(deltas)["mean"],
        "delta_summary": summarize(deltas),
        "bootstrap_95_ci": bootstrap_ci(deltas, seed=seed),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "pose_labels.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = sorted(
        metadata["samples"],
        key=lambda sample: int(sample["sample_id"]),
    )

    if not samples:
        raise RuntimeError("No dataset samples found.")

    records = []
    per_pose = []

    print("Disparity spatial-correlation scale sweep")
    print("=========================================")
    print("Poses:", len(samples))
    print("Repetitions per condition:", REPETITIONS)
    print("Conditions:", CONDITIONS)
    print("Correlation scales:", CORRELATION_SCALES_PX, "px")
    print("Disparity sigma:", DEFAULT_DISPARITY_SIGMA_PX, "px")
    print("Quantisation:", DEFAULT_DISPARITY_QUANTIZATION_PX, "px")
    print("Range dropout: disabled")
    print("Boundary dropout: disabled")
    print(
        "Expected trials:",
        len(samples) * len(CONDITIONS) * REPETITIONS,
    )

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
            raise FileNotFoundError(h5_path)

        with h5py.File(h5_path, "r") as f:
            depth_ideal = np.squeeze(f["depth"][:]).astype(np.float32)
            segmentation = np.squeeze(f["category_id_segmaps"][:])

        pose_result = {"sample_id": sid}

        for condition in CONDITIONS:
            pose_result[condition] = {
                "translation": [],
                "along": [],
                "cross": [],
                "axis": [],
                "cross_success": [],
                "translation_success": [],
            }

        for repetition in range(REPETITIONS):
            # The same seed is used for each pose/repetition across conditions.
            # Correlated scales share the same full-image white-noise draw;
            # independent mode instead draws independent values at valid pixels.
            seed = BASE_SEED + sample_index * 1009 + repetition

            for condition in CONDITIONS:
                if condition == "independent":
                    mode = "independent"
                    correlation_sigma = 8.0  # ignored in independent mode
                else:
                    mode = "correlated"
                    correlation_sigma = float(
                        condition.replace("correlated_", "").replace("px", "")
                    )

                depth_test, invalid_mask, edge_mask = apply_depth_noise(
                    depth_ideal,
                    segmentation,
                    seed=seed,
                    gaussian_noise=True,
                    range_missing=False,
                    boundary_failures=False,
                    disparity_noise_mode=mode,
                    correlation_sigma_px=correlation_sigma,
                )

                target_mask = segmentation == TARGET_CATEGORY
                original_count = int(np.count_nonzero(
                    target_mask
                    & np.isfinite(depth_ideal)
                    & (depth_ideal > 0)
                    & (depth_ideal < 1000)
                ))
                valid_count = int(np.count_nonzero(
                    target_mask
                    & np.isfinite(depth_test)
                    & (depth_test > 0)
                    & (depth_test < 1000)
                ))

                invalid_rate = (
                    1.0 - valid_count / original_count
                    if original_count > 0 else 1.0
                )

                record = {
                    "sample_id": sid,
                    "condition": condition,
                    "repetition": repetition,
                    "seed": seed,
                    "correlation_sigma_px": (
                        correlation_sigma if mode == "correlated" else 0.0
                    ),
                    "valid_target_points": valid_count,
                    "invalid_target_rate": invalid_rate,
                    "estimator_valid": False,
                    "translation_error_mm": "",
                    "along_axis_error_mm": "",
                    "cross_axis_error_mm": "",
                    "long_axis_error_deg": "",
                    "cross_axis_success": False,
                    "translation_success": False,
                    "error_message": "",
                }

                try:
                    errors = estimate_errors(
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
                        "cross_axis_success": cross_success,
                        "translation_success": translation_success,
                    })

                    pose_result[condition]["translation"].append(translation)
                    pose_result[condition]["along"].append(along)
                    pose_result[condition]["cross"].append(cross)
                    pose_result[condition]["axis"].append(axis)

                except Exception as exc:
                    record["error_message"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                # Failed estimates count as unsuccessful.
                pose_result[condition]["cross_success"].append(
                    bool(record["cross_axis_success"])
                )
                pose_result[condition]["translation_success"].append(
                    bool(record["translation_success"])
                )

                records.append(record)

        # Convert trial-level results into one observation per pose.
        for condition in CONDITIONS:
            values = pose_result[condition]

            for metric in ("translation", "along", "cross", "axis"):
                metric_values = values[metric]
                values[metric] = (
                    float(np.mean(metric_values))
                    if metric_values else None
                )

            values["coverage"] = (
                len(pose_result[condition]["cross_success"]) -
                sum(
                    1 for r in records[-len(CONDITIONS) * REPETITIONS:]
                    if r["sample_id"] == sid
                    and r["condition"] == condition
                    and not r["estimator_valid"]
                )
            ) / REPETITIONS

            values["cross_success_rate"] = float(
                np.mean(values["cross_success"])
            )
            values["translation_success_rate"] = float(
                np.mean(values["translation_success"])
            )

            del values["cross_success"]
            del values["translation_success"]

        per_pose.append(pose_result)

        if (sample_index + 1) % 10 == 0:
            print(
                f"Processed {sample_index + 1}/{len(samples)} poses "
                f"({len(records)} trials)"
            )

    # Trial-level CSV.
    trial_path = OUTPUT_DIR / "trial_results.csv"
    with open(trial_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Pose-level CSV.
    pose_rows = []
    for pose in per_pose:
        row = {"sample_id": pose["sample_id"]}
        for condition in CONDITIONS:
            for metric, value in pose[condition].items():
                row[f"{condition}_{metric}"] = value
        pose_rows.append(row)

    pose_path = OUTPUT_DIR / "per_pose_results.csv"
    with open(pose_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pose_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pose_rows)

    summary = {
        "experiment": "disparity_spatial_correlation_scale_sweep",
        "num_poses": len(samples),
        "repetitions_per_condition": REPETITIONS,
        "base_seed": BASE_SEED,
        "disparity_sigma_px": DEFAULT_DISPARITY_SIGMA_PX,
        "disparity_quantization_px": DEFAULT_DISPARITY_QUANTIZATION_PX,
        "correlation_scales_px": list(CORRELATION_SCALES_PX),
        "range_missing": False,
        "boundary_failures": False,
        "success_thresholds": {
            "cross_axis_mm": CROSS_AXIS_THRESHOLD_MM,
            "translation_mm": TRANSLATION_THRESHOLD_MM,
            "axis_deg": AXIS_THRESHOLD_DEG,
        },
        "conditions": {},
        "paired_comparisons_vs_independent": {},
    }

    for condition in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition]
        valid = [r for r in subset if r["estimator_valid"]]

        summary["conditions"][condition] = {
            "trials": len(subset),
            "valid_estimates": len(valid),
            "coverage": len(valid) / len(subset),
            "translation_error_mm": summarize([
                r["translation_error_mm"] for r in valid
            ]),
            "along_axis_error_mm": summarize([
                r["along_axis_error_mm"] for r in valid
            ]),
            "cross_axis_error_mm": summarize([
                r["cross_axis_error_mm"] for r in valid
            ]),
            "long_axis_error_deg": summarize([
                r["long_axis_error_deg"] for r in valid
            ]),
            "cross_axis_geometric_success_rate": float(np.mean([
                r["cross_axis_success"] for r in subset
            ])),
            "translation_geometric_success_rate": float(np.mean([
                r["translation_success"] for r in subset
            ])),
        }

    comparison_metrics = (
        "translation",
        "along",
        "cross",
        "axis",
        "coverage",
        "cross_success_rate",
        "translation_success_rate",
    )

    for condition_index, condition in enumerate(CONDITIONS[1:]):
        summary["paired_comparisons_vs_independent"][condition] = {}

        for metric_index, metric in enumerate(comparison_metrics):
            summary["paired_comparisons_vs_independent"][condition][metric] = (
                paired_comparison(
                    per_pose,
                    condition,
                    metric,
                    seed=BASE_SEED + 100 + condition_index * 10 + metric_index,
                )
            )

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCONDITION SUMMARY")
    print("=================")

    for condition, result in summary["conditions"].items():
        print(f"\n{condition}")
        print("  Coverage:", f"{100 * result['coverage']:.2f}%")
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

    print("\nPAIRED EFFECTS: EACH SCALE MINUS INDEPENDENT")
    print("============================================")

    for condition, metrics in summary[
        "paired_comparisons_vs_independent"
    ].items():
        print(f"\n{condition}")
        for metric in ("cross", "axis", "cross_success_rate"):
            result = metrics[metric]
            print(
                f"  {metric}: mean delta "
                f"{result['mean_delta_candidate_minus_independent']:.6f}; "
                f"95% CI {result['bootstrap_95_ci']}"
            )

    print("\nSaved:")
    print(trial_path)
    print(pose_path)
    print(summary_path)


if __name__ == "__main__":
    main()