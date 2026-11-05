import csv
import json
import h5py
import numpy as np

from pathlib import Path
from src.sensor.depth_noise import apply_depth_noise
from scripts.test_cad_end_estimator import (
    CATEGORY,
    LENGTH,
    CV_TO_BLENDER,
    analyse,
)

DATA = Path("outputs/randomized_pose_dataset")
OUT = Path("results/cad_end_estimator")

RANDOM_SEED = 20261104
NOISY_REPETITIONS = 3

# The nominal extrusion cross-section is 80 x 80 mm.
EXPECTED_FACE_MM = 80.0

CONDITIONS = {
    "ideal": {
        "gaussian_noise": False,
        "range_missing": False,
        "boundary_failures": False,
    },
    "disparity": {
        "gaussian_noise": True,
        "range_missing": False,
        "boundary_failures": False,
    },
    "dropout": {
        "gaussian_noise": False,
        "range_missing": True,
        "boundary_failures": True,
    },
    "full": {
        "gaussian_noise": True,
        "range_missing": True,
        "boundary_failures": True,
    },
}


def load_raw_sample(sample_id):
    path = DATA / f"{sample_id}.hdf5"

    with h5py.File(path, "r") as f:
        depth = np.squeeze(f["depth"][:]).astype(np.float32)
        segmentation = np.squeeze(
            f["category_id_segmaps"][:]
        )

    return depth, segmentation


def backproject_target(depth, segmentation, K, T_cw):
    mask = (
        (segmentation == CATEGORY)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(mask)
    z = depth[v, u].astype(float)

    if len(z) == 0:
        return np.empty((0, 3)), 0

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    points_world = (
        T_cw @ CV_TO_BLENDER @ points_cv.T
    ).T[:, :3]

    return points_world, len(z)


def select_end_plane(planes):
    """
    Select a plausible end-face plane.

    The two largest in-plane dimensions should be reasonably
    consistent with the nominal 80 x 80 mm cross-section.
    """

    candidates = []

    for plane in planes:
        dims = np.sort(plane["dimensions_mm"])

        thickness = dims[0]
        face_dims = dims[1:]

        # Reject geometrically implausible candidate planes.
        if thickness > 12:
            continue

        if np.any(face_dims < 30) or np.any(face_dims > 110):
            continue

        dimension_error = np.sum(
            np.abs(face_dims - EXPECTED_FACE_MM)
        )

        residual_penalty = plane["p95_residual_mm"]

        # Prefer plausible cross-sections and low-residual planes.
        score = dimension_error + 2.0 * residual_penalty

        candidates.append((score, plane))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]


def evaluate_sample(
    sample,
    K,
    condition,
    repetition,
    min_total_candidates=20,
    min_cluster_points=20,
):
    sid = int(sample["sample_id"])

    depth, segmentation = load_raw_sample(sid)

    # Corresponding conditions use the same pose/repetition seed.
    seed = RANDOM_SEED + sid * 100 + repetition

    params = CONDITIONS[condition]

    corrupted_depth, invalid_mask, edge_mask = apply_depth_noise(
        depth,
        segmentation,
        seed=seed,
        **params,
    )

    T_cw = np.asarray(
        sample["camera_to_world_blender"],
        dtype=float,
    )

    points, target_points = backproject_target(
        corrupted_depth,
        segmentation,
        K,
        T_cw,
    )

    original_target_mask = (
        (segmentation == CATEGORY)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    original_target_count = np.count_nonzero(
        original_target_mask
    )

    invalid_target_count = np.count_nonzero(
        invalid_mask & original_target_mask
    )

    invalid_fraction = (
        invalid_target_count / original_target_count
        if original_target_count else np.nan
    )

    result = {
        "sample_id": sid,
        "condition": condition,
        "repetition": repetition,
        "seed": seed,
        "target_points": len(points),
        "invalid_target_fraction": invalid_fraction,
        "axis_error_deg": np.nan,
        "translation_error_mm": np.nan,
        "candidate_planes": 0,
        "selected_plane_points": 0,
        "plane_median_residual_mm": np.nan,
        "plane_p95_residual_mm": np.nan,
        "face_dimension_1_mm": np.nan,
        "face_dimension_2_mm": np.nan,
        "success": False,
        "failure_reason": "",
        "min_total_candidates": min_total_candidates,
        "min_cluster_points": min_cluster_points,
        "end_normal_candidates": 0,
    }

    if len(points) < 30:
        result["failure_reason"] = "insufficient_target_points"
        return result

    try:
        axis, n_candidates, planes = analyse(
            points,
            T_cw[:3, 3],
            min_candidates=min_total_candidates,
            min_cluster_points=min_cluster_points,
        )
    except Exception as exc:
        result["failure_reason"] = (
            f"estimator_exception: {type(exc).__name__}: {exc}"
        )
        return result

    result["end_normal_candidates"] = int(n_candidates)
    result["candidate_planes"] = len(planes)

    if n_candidates < min_total_candidates:
        result["failure_reason"] = "insufficient_end_normal_candidates"
        return result

    selected = select_end_plane(planes)

    if selected is None:
        result["failure_reason"] = "no_plausible_end_plane"
        return result

    T_gt = np.asarray(
        sample["target_to_world"],
        dtype=float,
    )

    gt_centre = T_gt[:3, 3]

    # Symmetry-aware longitudinal-axis error.
    gt_axis = T_gt[:2, 0]
    gt_axis /= np.linalg.norm(gt_axis)

    axis_error = np.degrees(
        np.arccos(
            np.clip(abs(np.dot(axis, gt_axis)), -1.0, 1.0)
        )
    )

    # The plane normal points towards the camera.
    normal_xy = selected["normal"][:2].copy()
    normal_xy /= np.linalg.norm(normal_xy)

    estimated_centre = (
        selected["centroid"][:2]
        - 0.5 * LENGTH * normal_xy
    )

    translation_error = 1000 * np.linalg.norm(
        estimated_centre - gt_centre[:2]
    )

    dims = np.sort(selected["dimensions_mm"])

    result.update({
        "axis_error_deg": float(axis_error),
        "translation_error_mm": float(translation_error),
        "selected_plane_points": int(selected["count"]),
        "plane_median_residual_mm": selected["median_residual_mm"],
        "plane_p95_residual_mm": selected["p95_residual_mm"],
        "face_dimension_1_mm": float(dims[-2]),
        "face_dimension_2_mm": float(dims[-1]),
        "success": True,
    })

    return result


def summarise(rows):
    summary = {}

    for condition in CONDITIONS:
        subset = [
            r for r in rows
            if r["condition"] == condition
        ]

        valid = [
            r for r in subset
            if r["success"]
        ]

        translations = np.array([
            r["translation_error_mm"] for r in valid
        ])

        angles = np.array([
            r["axis_error_deg"] for r in valid
        ])

        def percentile(values, q):
            return (
                float(np.percentile(values, q))
                if len(values) else None
            )

        # Failed estimates count as failures, not successes.
        success_5mm_5deg = sum(
            r["success"]
            and r["translation_error_mm"] <= 5
            and r["axis_error_deg"] <= 5
            for r in subset
        )

        success_10mm_5deg = sum(
            r["success"]
            and r["translation_error_mm"] <= 10
            and r["axis_error_deg"] <= 5
            for r in subset
        )

        summary[condition] = {
            "trials": len(subset),
            "successful_estimates": len(valid),
            "estimation_failure_rate": (
                1 - len(valid) / len(subset)
                if subset else None
            ),
            "translation_median_mm": percentile(translations, 50),
            "translation_mean_mm": (
                float(np.mean(translations))
                if len(translations) else None
            ),
            "translation_p95_mm": percentile(translations, 95),
            "axis_median_deg": percentile(angles, 50),
            "axis_mean_deg": (
                float(np.mean(angles))
                if len(angles) else None
            ),
            "axis_p95_deg": percentile(angles, 95),
            "success_5mm_5deg": success_5mm_5deg,
            "success_5mm_5deg_fraction": (
                success_5mm_5deg / len(subset)
                if subset else None
            ),
            "success_10mm_5deg": success_10mm_5deg,
            "success_10mm_5deg_fraction": (
                success_10mm_5deg / len(subset)
                if subset else None
            ),
        }

    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA / "pose_labels.json") as f:
        metadata = json.load(f)

    K = np.asarray(metadata["intrinsics"], dtype=float)
    samples = metadata["samples"]

    rows = []

    total = 0

    for condition in CONDITIONS:
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
                )

                rows.append(result)
                total += 1

                if total % 50 == 0:
                    print(
                        f"Completed {total} trials; "
                        f"latest: {condition}, "
                        f"sample {result['sample_id']}, "
                        f"success={result['success']}"
                    )

    csv_path = OUT / "per_trial_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSUMMARY")

    for condition, values in summary.items():
        print(f"\n{condition.upper()}")
        print(
            "Successful estimates:",
            f"{values['successful_estimates']}/{values['trials']}",
        )
        print(
            "Translation median / P95:",
            values["translation_median_mm"],
            "/",
            values["translation_p95_mm"],
            "mm",
        )
        print(
            "Axis median / P95:",
            values["axis_median_deg"],
            "/",
            values["axis_p95_deg"],
            "deg",
        )
        print(
            "Success at 5 mm / 5 deg:",
            values["success_5mm_5deg_fraction"],
        )
        print(
            "Success at 10 mm / 5 deg:",
            values["success_10mm_5deg_fraction"],
        )

    print("\nSaved:", csv_path)
    print("Saved:", OUT / "summary.json")


if __name__ == "__main__":
    main()