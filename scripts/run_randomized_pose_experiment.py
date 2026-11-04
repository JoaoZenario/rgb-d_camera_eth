import csv
import glob
import json
import os
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import apply_depth_noise
from src.experiments.obb_pose import (
    estimate_obb_pose,
    evaluate_pose,
)


DATASET_DIR = Path("outputs/randomized_pose_dataset")
OUTPUT_DIR = Path("outputs/randomized_pose_experiment")

TARGET_CATEGORY = 4

BASE_SEED = 20261104
CORRUPTED_REPETITIONS = 5

POSITION_TOLERANCE_MM = 5.0
AXIS_TOLERANCE_DEG = 5.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_intrinsics(value):
    """Read K from either a matrix or a metadata dictionary."""

    if isinstance(value, dict):
        if "K" in value:
            return np.asarray(value["K"], dtype=float)

        if "matrix" in value:
            return np.asarray(value["matrix"], dtype=float)

        return np.array([
            [value["fx"], 0.0, value["cx"]],
            [0.0, value["fy"], value["cy"]],
            [0.0, 0.0, 1.0],
        ])

    return np.asarray(value, dtype=float)


def summarise(rows):
    """Summarise trials, counting estimator failures as grasp failures."""

    summary = {}

    for condition in ("ideal", "full_corruption"):
        subset = [r for r in rows if r["condition"] == condition]

        successful_estimates = [
            r for r in subset
            if r["estimation_valid"]
        ]

        translations = np.array([
            r["translation_error_mm"]
            for r in successful_estimates
        ], dtype=float)

        axes = np.array([
            r["long_axis_error_deg"]
            for r in successful_estimates
        ], dtype=float)

        successes = np.array([
            r["grasp_success"] for r in subset
        ], dtype=bool)

        invalid_rates = np.array([
            r["invalid_target_rate_pct"] for r in subset
        ], dtype=float)

        def error_stats(values):
            if len(values) == 0:
                return {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "p95": None,
                }

            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1))
                    if len(values) > 1 else 0.0,
                "p95": float(np.percentile(values, 95)),
            }

        summary[condition] = {
            "trials": len(subset),
            "valid_estimates": len(successful_estimates),
            "estimation_failures": (
                len(subset) - len(successful_estimates)
            ),
            "grasp_successes": int(successes.sum()),
            "grasp_success_rate_pct": float(
                100.0 * successes.mean()
            ),
            "mean_invalid_target_rate_pct": float(
                np.mean(invalid_rates)
            ),
            "translation_error_mm": error_stats(translations),
            "long_axis_error_deg": error_stats(axes),
        }

    return summary


def main():

    metadata_path = DATASET_DIR / "pose_labels.json"

    with open(metadata_path) as f:
        meta = json.load(f)

    K = get_intrinsics(meta["intrinsics"])

    files = sorted(
        glob.glob(str(DATASET_DIR / "*.hdf5")),
        key=lambda p: int(
            os.path.splitext(os.path.basename(p))[0]
        ),
    )

    samples = meta["samples"]

    if len(files) != len(samples):
        raise RuntimeError(
            f"Found {len(files)} HDF5 files but "
            f"{len(samples)} metadata samples."
        )

    rows = []

    print("Randomized PCA-OBB experiment")
    print("=============================")
    print("Samples:", len(samples))
    print("Corrupted repetitions per sample:", CORRUPTED_REPETITIONS)
    print("Expected trials:", len(samples) * (
        1 + CORRUPTED_REPETITIONS
    ))

    for sample_index, (path, sample) in enumerate(zip(files, samples)):

        with h5py.File(path, "r") as f:
            depth_ideal = np.squeeze(
                f["depth"][:]
            ).astype(np.float32)

            segmentation = np.squeeze(
                f["category_id_segmaps"][:]
            )

        T_target_gt = np.asarray(
            sample["target_to_world"], dtype=float
        )

        T_camera = np.asarray(
            sample["camera_to_world_blender"], dtype=float
        )

        target_mask = segmentation == TARGET_CATEGORY

        ideal_valid = (
            target_mask
            & np.isfinite(depth_ideal)
            & (depth_ideal > 0)
            & (depth_ideal < 1000)
        )

        ideal_target_count = int(ideal_valid.sum())

        if ideal_target_count == 0:
            raise RuntimeError(
                f"Sample {sample_index} has no valid ideal target depth."
            )

        conditions = [("ideal", -1, depth_ideal)]

        for repetition in range(CORRUPTED_REPETITIONS):

            # Unique deterministic seed for each sample/repetition.
            seed = (
                BASE_SEED
                + sample_index * 1009
                + repetition
            )

            corrupted_depth, _, _ = apply_depth_noise(
                depth_ideal,
                segmentation,
                seed=seed,
                gaussian_noise=True,
                range_missing=True,
                boundary_failures=True,
            )

            conditions.append((
                "full_corruption",
                seed,
                corrupted_depth,
            ))

        for condition, seed, depth in conditions:

            valid_target = (
                target_mask
                & np.isfinite(depth)
                & (depth > 0)
                & (depth < 1000)
            )

            point_count = int(valid_target.sum())

            invalid_rate = (
                100.0
                * (ideal_target_count - point_count)
                / ideal_target_count
            )

            row = {
                "sample_id": sample_index,
                "condition": condition,
                "seed": seed,
                "ideal_target_pixels": ideal_target_count,
                "valid_target_points": point_count,
                "invalid_target_rate_pct": invalid_rate,
                "estimation_valid": False,
                "translation_error_mm": "",
                "long_axis_error_deg": "",
                "grasp_success": False,
                "error_message": "",
            }

            try:
                estimate = estimate_obb_pose(
                    depth=depth,
                    segmentation=segmentation,
                    K=K,
                    camera_to_world=T_camera,
                    target_category=TARGET_CATEGORY,
                )

                errors = evaluate_pose(
                    estimate,
                    T_target_gt,
                )

                success = (
                    errors["translation_error_mm"]
                    <= POSITION_TOLERANCE_MM
                    and errors["long_axis_error_deg"]
                    <= AXIS_TOLERANCE_DEG
                )

                row.update({
                    "estimation_valid": True,
                    "translation_error_mm": (
                        errors["translation_error_mm"]
                    ),
                    "long_axis_error_deg": (
                        errors["long_axis_error_deg"]
                    ),
                    "grasp_success": bool(success),
                })

            except Exception as exc:
                row["error_message"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            rows.append(row)

        if (sample_index + 1) % 10 == 0:
            print(
                f"Processed {sample_index + 1}/{len(samples)} "
                f"samples; trials recorded: {len(rows)}"
            )

    # Save every individual trial, including failed estimates.
    csv_path = OUTPUT_DIR / "trial_results.csv"

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)

    summary_path = OUTPUT_DIR / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT SUMMARY")
    print("==================")

    for condition, stats in summary.items():
        print(f"\n{condition}")
        print("Trials:", stats["trials"])
        print("Valid estimates:", stats["valid_estimates"])
        print("Estimation failures:", stats["estimation_failures"])
        print(
            "Grasp success rate:",
            f"{stats['grasp_success_rate_pct']:.1f}%"
        )
        print(
            "Mean invalid target depth:",
            f"{stats['mean_invalid_target_rate_pct']:.2f}%"
        )

        print(
            "Translation error (mm):",
            stats["translation_error_mm"]
        )

        print(
            "Long-axis error (deg):",
            stats["long_axis_error_deg"]
        )

    print("\nSaved:", csv_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()