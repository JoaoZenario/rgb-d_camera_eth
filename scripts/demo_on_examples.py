"""Run a lightweight smoke test on the three bundled RGB-D example frames.

This script does not require Blender, BlenderProc, or Open3D. It exercises the
provided depth-corruption model and PCA-OBB estimator on a small, fixed subset
of the labelled synthetic dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.experiments.obb_pose import estimate_obb_pose, evaluate_pose
from src.sensor.depth_noise import apply_depth_noise


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "randomized_pose_dataset"
FRAME_DIR = EXAMPLE_DIR / "sample_frames"
METADATA_PATH = EXAMPLE_DIR / "pose_labels.json"
TARGET_CATEGORY = 4
BASE_SEED = 20261104


def load_intrinsics(value: object) -> np.ndarray:
    """Accept the matrix or dictionary formats used by the dataset metadata."""
    if isinstance(value, dict):
        if "K" in value:
            value = value["K"]
        elif "matrix" in value:
            value = value["matrix"]
        else:
            value = [
                [value["fx"], 0.0, value["cx"]],
                [0.0, value["fy"], value["cy"]],
                [0.0, 0.0, 1.0],
            ]
    K = np.asarray(value, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 camera intrinsics, got {K.shape}")
    return K


def load_frame(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    frame_path = FRAME_DIR / f"{sample['frame']}.hdf5"
    if not frame_path.is_file():
        raise FileNotFoundError(f"Example frame not found: {frame_path}")

    with h5py.File(frame_path, "r") as handle:
        depth = np.asarray(handle["depth"][:], dtype=np.float32).squeeze()
        segmentation = np.asarray(
            handle["category_id_segmaps"][:]
        ).squeeze()

    if depth.ndim != 2 or segmentation.shape != depth.shape:
        raise ValueError(
            f"Unexpected frame shapes: depth={depth.shape}, "
            f"segmentation={segmentation.shape}"
        )
    return depth, segmentation


def estimate_and_evaluate(
    depth: np.ndarray,
    segmentation: np.ndarray,
    K: np.ndarray,
    camera_to_world: np.ndarray,
    target_to_world: np.ndarray,
) -> dict:
    estimate = estimate_obb_pose(
        depth=depth,
        segmentation=segmentation,
        K=K,
        camera_to_world=camera_to_world,
        target_category=TARGET_CATEGORY,
    )
    errors = evaluate_pose(estimate, target_to_world)
    errors["point_count"] = estimate["point_count"]
    return errors


def format_result(result: dict) -> str:
    return (
        f"n={result['point_count']:4d} | "
        f"translation={result['translation_error_mm']:7.2f} mm "
        f"(cross={result['cross_axis_error_mm']:7.2f}, "
        f"along={result['along_axis_error_mm']:6.2f}) | "
        f"axis={result['long_axis_error_deg']:5.2f} deg"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test depth corruption and OBB estimation on bundled examples."
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 100, 199],
        help="Sample IDs to run (default: 0 100 199).",
    )
    args = parser.parse_args()

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Example metadata not found: {METADATA_PATH}. "
            "Run this command from the repository checkout."
        )

    with METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    samples_by_id = {
        int(sample["sample_id"]): sample for sample in metadata["samples"]
    }
    K = load_intrinsics(metadata["intrinsics"])

    print("RGB-D example smoke test (no Blender / BlenderProc / Open3D)")
    print("=" * 66)
    print("Estimator: PCA-oriented bounding box on ground-truth target segmentation")
    print("Corruption: default independent-disparity + range/boundary dropout model")
    print("Examples:", ", ".join(str(i) for i in args.samples))
    print("Note: this small demo is not a reproduction of the 200-pose experiment.\n")

    completed = 0
    failures = 0

    for sample_id in args.samples:
        if sample_id not in samples_by_id:
            print(f"Sample {sample_id}: NOT FOUND in pose metadata")
            failures += 1
            continue

        sample = samples_by_id[sample_id]
        depth, segmentation = load_frame(sample)
        camera_to_world = np.asarray(
            sample["camera_to_world_blender"], dtype=float
        )
        target_to_world = np.asarray(sample["target_to_world"], dtype=float)

        target_mask = segmentation == TARGET_CATEGORY
        ideal_valid = (
            target_mask & np.isfinite(depth) & (depth > 0) & (depth < 1000)
        )
        if not np.any(ideal_valid):
            print(f"Sample {sample_id}: no valid target depth pixels")
            failures += 1
            continue

        # Match the deterministic seed convention used by the full OBB run.
        seed = BASE_SEED + sample_id * 1009
        corrupted_depth, invalid_mask, _ = apply_depth_noise(
            depth,
            segmentation,
            seed=seed,
            gaussian_noise=True,
            range_missing=True,
            boundary_failures=True,
        )

        invalid_target_pct = (
            100.0 * np.count_nonzero(invalid_mask & target_mask)
            / np.count_nonzero(ideal_valid)
        )

        print(f"Sample {sample_id} (frame {sample['frame']}):")
        try:
            ideal_result = estimate_and_evaluate(
                depth, segmentation, K, camera_to_world, target_to_world
            )
            print("  Ideal:      " + format_result(ideal_result))
        except Exception as exc:  # keep the demo moving if one sample fails
            ideal_result = None
            print(f"  Ideal:      ESTIMATION FAILED ({type(exc).__name__}: {exc})")
            failures += 1

        try:
            corrupted_result = estimate_and_evaluate(
                corrupted_depth,
                segmentation,
                K,
                camera_to_world,
                target_to_world,
            )
            print("  Corrupted:  " + format_result(corrupted_result))
        except Exception as exc:
            corrupted_result = None
            print(
                f"  Corrupted:  ESTIMATION FAILED ({type(exc).__name__}: {exc})"
            )
            failures += 1

        print(f"  Target depth invalidated: {invalid_target_pct:.2f}%")
        if ideal_result is not None and corrupted_result is not None:
            delta = (
                corrupted_result["translation_error_mm"]
                - ideal_result["translation_error_mm"]
            )
            print(f"  Change in translation error: {delta:+.2f} mm")
        print()
        completed += 1

    print(f"Completed {completed}/{len(args.samples)} requested examples.")
    if failures:
        print(f"Estimator or input issues: {failures} (see messages above).")
    return 0 if completed == len(args.samples) and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())