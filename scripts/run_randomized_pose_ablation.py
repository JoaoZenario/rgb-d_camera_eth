import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import apply_depth_noise
from src.experiments.obb_pose import (
    estimate_obb_pose,
    evaluate_pose,
)

DATA_DIR = Path("outputs/randomized_pose_dataset")
OUTPUT_DIR = Path("outputs/randomized_pose_ablation")

BASE_SEED = 20261104
REPETITIONS = 5
TARGET_CATEGORY = 4

TRANSLATION_THRESHOLD_MM = 5.0
AXIS_THRESHOLD_DEG = 5.0

CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])

CONDITIONS = {
    "ideal": (False, False, False),
    "disparity_only": (True, False, False),
    "dropout_only": (False, True, True),
    "full_corruption": (True, True, True),
}


def percentile(values, p):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return float(np.percentile(values, p))


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
        "p95": percentile(values, 95),
    }


def bootstrap_ci(values, seed=20261105, repetitions=10000):
    """Bootstrap CI over independent pose-level observations."""
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


def visibility_diagnostics(depth, segmentation, K, T_cw, T_wo):
    """Measure visible longitudinal coverage in target coordinates."""
    valid = (
        (segmentation == TARGET_CATEGORY)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(valid)

    if len(u) < 100:
        return {
            "axis_view_angle_deg": None,
            "visible_span_p1_p99_mm": None,
            "visible_midpoint_offset_mm": None,
        }

    z = depth[v, u].astype(float)

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    points_world = (T_cw @ (CV_TO_BLENDER @ points_cv.T)).T

    points_local = (
        np.linalg.inv(T_wo) @ points_world.T
    ).T[:, :3]

    x_mm = points_local[:, 0] * 1000.0

    lo, hi = np.percentile(x_mm, [1, 99])

    camera_view = T_cw[:3, :3] @ np.array([0.0, 0.0, -1.0])
    camera_view /= np.linalg.norm(camera_view)

    long_axis = T_wo[:3, 0]
    long_axis /= np.linalg.norm(long_axis)

    cosine = np.clip(abs(np.dot(long_axis, camera_view)), 0.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return {
        "axis_view_angle_deg": float(angle),
        "visible_span_p1_p99_mm": float(hi - lo),
        "visible_midpoint_offset_mm": float((hi + lo) / 2.0),
    }


def run_estimation(depth, segmentation, K, T_cw, T_wo):
    estimate = estimate_obb_pose(
        depth=depth,
        segmentation=segmentation,
        K=K,
        camera_to_world=T_cw,
        target_category=TARGET_CATEGORY,
    )

    errors = evaluate_pose(estimate, T_wo)

    success = (
        errors["translation_error_mm"] <= TRANSLATION_THRESHOLD_MM
        and errors["long_axis_error_deg"] <= AXIS_THRESHOLD_DEG
    )

    return errors, success


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = sorted(metadata["samples"], key=lambda s: int(s["sample_id"]))

    expected_files = len(samples)

    print("Randomized paired pose ablation")
    print("===============================")
    print("Samples:", expected_files)
    print("Corruption repetitions:", REPETITIONS)
    print("Conditions:", list(CONDITIONS))
    print(
        "Expected trials:",
        expected_files * (1 + 3 * REPETITIONS),
    )

    records = []
    ideal_by_sample = {}
    sample_condition_errors = {}

    for sample_index, sample in enumerate(samples):
        sid = int(sample["sample_id"])

        T_cw = np.asarray(sample["camera_to_world_blender"], dtype=float)
        T_wo = np.asarray(sample["target_to_world"], dtype=float)

        h5_path = DATA_DIR / f"{sid}.hdf5"

        with h5py.File(h5_path, "r") as f:
            depth_ideal = np.squeeze(f["depth"][:]).astype(np.float32)
            segmentation = np.squeeze(f["category_id_segmaps"][:])

        target_mask = segmentation == TARGET_CATEGORY
        original_target_count = int(np.count_nonzero(
            target_mask & np.isfinite(depth_ideal)
            & (depth_ideal > 0) & (depth_ideal < 1000)
        ))

        visibility = visibility_diagnostics(
            depth_ideal, segmentation, K, T_cw, T_wo
        )

        sample_condition_errors[sid] = {}

        for condition, flags in CONDITIONS.items():
            gaussian_noise, range_missing, boundary_failures = flags

            repetitions = 1 if condition == "ideal" else REPETITIONS

            for repetition in range(repetitions):
                seed = BASE_SEED + sample_index * 1009 + repetition

                if condition == "ideal":
                    depth_test = depth_ideal.copy()
                else:
                    depth_test, invalid_mask, edge_mask = apply_depth_noise(
                        depth_ideal,
                        segmentation,
                        seed=seed,
                        gaussian_noise=gaussian_noise,
                        range_missing=range_missing,
                        boundary_failures=boundary_failures,
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
                    if original_target_count else 1.0
                )

                record = {
                    "sample_id": sid,
                    "condition": condition,
                    "repetition": repetition,
                    "seed": seed,
                    "valid_target_points": valid_count,
                    "invalid_target_rate": invalid_rate,
                    **visibility,
                    "estimator_valid": False,
                    "translation_error_mm": "",
                    "along_axis_error_mm": "",
                    "cross_axis_error_mm": "",
                    "long_axis_error_deg": "",
                    "grasp_success": False,
                    "error_message": "",
                }

                try:
                    errors, success = run_estimation(
                        depth_test, segmentation, K, T_cw, T_wo
                    )

                    record["estimator_valid"] = True
                    record["translation_error_mm"] = errors[
                        "translation_error_mm"
                    ]
                    record["along_axis_error_mm"] = errors[
                        "along_axis_error_mm"
                    ]
                    record["cross_axis_error_mm"] = errors[
                        "cross_axis_error_mm"
                    ]
                    record["long_axis_error_deg"] = errors[
                        "long_axis_error_deg"
                    ]
                    record["grasp_success"] = success

                except Exception as exc:
                    record["error_message"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                records.append(record)

                if record["estimator_valid"]:
                    sample_condition_errors[sid].setdefault(
                        condition, []
                    ).append({
                        "translation": record["translation_error_mm"],
                        "along": record["along_axis_error_mm"],
                        "cross": record["cross_axis_error_mm"],
                        "axis": record["long_axis_error_deg"],
                        "success": record["grasp_success"],
                    })

                if condition == "ideal":
                    ideal_by_sample[sid] = record

        if (sample_index + 1) % 10 == 0:
            print(
                f"Processed {sample_index + 1}/{expected_files}; "
                f"trials recorded: {len(records)}"
            )

    # Save all individual trials.
    csv_path = OUTPUT_DIR / "trial_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "num_samples": expected_files,
        "corruption_repetitions_per_sample": REPETITIONS,
        "grasp_translation_threshold_mm": TRANSLATION_THRESHOLD_MM,
        "grasp_axis_threshold_deg": AXIS_THRESHOLD_DEG,
        "conditions": {},
        "paired_comparisons_vs_ideal": {},
    }

    # Trial-level condition summaries.
    for condition in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition]
        valid = [r for r in subset if r["estimator_valid"]]

        translations = [float(r["translation_error_mm"]) for r in valid]
        along_errors = [float(r["along_axis_error_mm"]) for r in valid]
        cross_errors = [float(r["cross_axis_error_mm"]) for r in valid]
        axes = [float(r["long_axis_error_deg"]) for r in valid]

        summary["conditions"][condition] = {
            "trials": len(subset),
            "valid_estimates": len(valid),
            "estimation_failures": len(subset) - len(valid),
            "grasp_successes": sum(bool(r["grasp_success"]) for r in subset),
            "grasp_success_rate": (
                sum(bool(r["grasp_success"]) for r in subset) / len(subset)
                if subset else None
            ),
            "mean_invalid_target_rate": (
                float(np.mean([r["invalid_target_rate"] for r in subset]))
                if subset else None
            ),
            "translation_error_mm": summarize(translations),
            "along_axis_error_mm": summarize(along_errors),
            "cross_axis_error_mm": summarize(cross_errors),
            "long_axis_error_deg": summarize(axes),
        }

    # Paired comparison: average repeated trials within each pose first.
    ideal_samples = {
        sid: values["ideal"][0]
        for sid, values in sample_condition_errors.items()
        if "ideal" in values and len(values["ideal"]) == 1
    }

    for condition in ["disparity_only", "dropout_only", "full_corruption"]:
        paired_translation_delta = []
        paired_along_delta = []
        paired_cross_delta = []
        paired_axis_delta = []
        paired_success_delta = []

        for sid, ideal in ideal_samples.items():
            corrupted = sample_condition_errors[sid].get(condition, [])

            if len(corrupted) != REPETITIONS:
                continue

            paired_translation_delta.append(
                np.mean([x["translation"] for x in corrupted])
                - ideal["translation"]
            )

            paired_along_delta.append(
                np.mean([x["along"] for x in corrupted])
                - ideal["along"]
            )

            paired_cross_delta.append(
                np.mean([x["cross"] for x in corrupted])
                - ideal["cross"]
            )

            paired_axis_delta.append(
                np.mean([x["axis"] for x in corrupted])
                - ideal["axis"]
            )

            paired_success_delta.append(
                np.mean([float(x["success"]) for x in corrupted])
                - float(ideal["success"])
            )

        summary["paired_comparisons_vs_ideal"][condition] = {
            "paired_samples": len(paired_translation_delta),
            "translation_delta_mm": summarize(paired_translation_delta),
            "translation_delta_bootstrap_95_ci_mm": bootstrap_ci(
                paired_translation_delta, seed=BASE_SEED + 1
            ),
            "along_axis_delta_mm": summarize(paired_along_delta),
            "along_axis_delta_bootstrap_95_ci_mm": bootstrap_ci(
                paired_along_delta, seed=BASE_SEED + 4
            ),
            "cross_axis_delta_mm": summarize(paired_cross_delta),
            "cross_axis_delta_bootstrap_95_ci_mm": bootstrap_ci(
                paired_cross_delta, seed=BASE_SEED + 5
            ),
            "axis_delta_deg": summarize(paired_axis_delta),
            "axis_delta_bootstrap_95_ci_deg": bootstrap_ci(
                paired_axis_delta, seed=BASE_SEED + 2
            ),
            "success_rate_delta": summarize(paired_success_delta),
            "success_rate_delta_bootstrap_95_ci": bootstrap_ci(
                paired_success_delta, seed=BASE_SEED + 3
            ),
        }

    json_path = OUTPUT_DIR / "summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCONDITION SUMMARY")
    print("=================")

    for condition, result in summary["conditions"].items():
        print(f"\n{condition}")
        print("  Trials:", result["trials"])
        print("  Valid estimates:", result["valid_estimates"])
        print(
            "  Grasp success:",
            f"{100 * result['grasp_success_rate']:.2f}%"
        )
        print(
            "  Translation median:",
            result["translation_error_mm"]["median"],
            "mm"
        )
        print(
            "  Translation P95:",
            result["translation_error_mm"]["p95"],
            "mm"
        )
        print(
            "  Long-axis median:",
            result["long_axis_error_deg"]["median"],
            "deg"
        )

    print("\nPAIRED CORRUPTION EFFECTS")
    print("=========================")

    for condition, result in summary["paired_comparisons_vs_ideal"].items():
        delta = result["translation_delta_mm"]
        ci = result["translation_delta_bootstrap_95_ci_mm"]

        print(f"\n{condition}")
        print("  Paired samples:", result["paired_samples"])
        print("  Mean translation change:", delta["mean"], "mm")
        print("  Bootstrap 95% CI:", ci)
        print(
            "  Mean axis change:",
            result["axis_delta_deg"]["mean"],
            "deg"
        )
        print(
            "  Mean success-rate change:",
            result["success_rate_delta"]["mean"]
        )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()