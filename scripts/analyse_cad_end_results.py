import json
import numpy as np
import pandas as pd

from pathlib import Path

INPUT = Path("results/cad_end_estimator/per_trial_results.csv")
OUT = Path("results/cad_end_estimator")
SEED = 20261105
BOOTSTRAP_REPS = 10000

CONDITIONS = ["ideal", "disparity", "dropout", "full"]
COMPARISONS = ["disparity", "dropout", "full"]


def parse_bool(series):
    """Robustly parse CSV booleans, including string-valued True/False."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def bootstrap_mean_ci(values, rng):
    """Percentile bootstrap CI for a mean, resampling independent poses."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPS, len(values))
    )
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def paired_bootstrap(differences, rng):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) == 0:
        return np.nan, np.nan, np.nan

    ci_low, ci_high = bootstrap_mean_ci(differences, rng)
    return float(differences.mean()), float(ci_low), float(ci_high)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "sample_id", "condition", "translation_error_mm",
        "axis_error_deg", "success"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df["success_bool"] = parse_bool(df["success"])
    df["valid"] = (
        df["translation_error_mm"].notna()
        & df["axis_error_deg"].notna()
    )

    # Geometric success requires a returned pose within the stated
    # translation and axis tolerances. Failed estimates count as failures.
    df["geometric_success_5mm_5deg"] = (
        df["valid"]
        & (df["translation_error_mm"] <= 5.0)
        & (df["axis_error_deg"] <= 5.0)
    )
    df["geometric_success_10mm_5deg"] = (
        df["valid"]
        & (df["translation_error_mm"] <= 10.0)
        & (df["axis_error_deg"] <= 5.0)
    )

    # Aggregate repetitions within each independent pose and condition.
    per_pose = (
        df.groupby(["sample_id", "condition"], as_index=False)
        .agg(
            coverage=("valid", "mean"),
            geometric_success_5mm_5deg=(
                "geometric_success_5mm_5deg", "mean"
            ),
            geometric_success_10mm_5deg=(
                "geometric_success_10mm_5deg", "mean"
            ),
            translation_error_mm=("translation_error_mm", "mean"),
            axis_error_deg=("axis_error_deg", "mean"),
            valid_repetitions=("valid", "sum"),
            total_repetitions=("valid", "size"),
        )
    )

    # Ensure every condition is represented for every pose.
    pose_ids = sorted(df["sample_id"].unique())
    if len(pose_ids) != 200:
        print(f"WARNING: expected 200 poses, found {len(pose_ids)}")

    summary_rows = []
    for condition in CONDITIONS:
        rows = df[df["condition"] == condition]
        pose_rows = per_pose[per_pose["condition"] == condition]

        valid_rows = rows[rows["valid"]]

        summary_rows.append({
            "condition": condition,
            "num_poses": pose_rows["sample_id"].nunique(),
            "trials": len(rows),
            "coverage": float(pose_rows["coverage"].mean()),
            "geometric_success_5mm_5deg_rate": float(
                pose_rows["geometric_success_5mm_5deg"].mean()
            ),
            "geometric_success_10mm_5deg_rate": float(
                pose_rows["geometric_success_10mm_5deg"].mean()
            ),
            "valid_estimates": int(rows["valid"].sum()),
            "translation_median_mm": float(
                valid_rows["translation_error_mm"].median()
            ) if len(valid_rows) else np.nan,
            "translation_p95_mm": float(
                valid_rows["translation_error_mm"].quantile(0.95)
            ) if len(valid_rows) else np.nan,
            "axis_median_deg": float(
                valid_rows["axis_error_deg"].median()
            ) if len(valid_rows) else np.nan,
            "axis_p95_deg": float(
                valid_rows["axis_error_deg"].quantile(0.95)
            ) if len(valid_rows) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)

    # Paired comparisons: resample poses, never individual repetitions.
    wide = per_pose.pivot(
        index="sample_id",
        columns="condition",
        values=[
            "coverage",
            "geometric_success_5mm_5deg",
            "geometric_success_10mm_5deg",
            "translation_error_mm",
            "axis_error_deg",
        ],
    )

    rng = np.random.default_rng(SEED)
    comparison_rows = []

    for condition in COMPARISONS:
        # Coverage and geometric success: all poses are included.
        for metric in [
            "coverage",
            "geometric_success_5mm_5deg",
            "geometric_success_10mm_5deg",
        ]:
            ideal_values = wide[(metric, "ideal")]
            condition_values = wide[(metric, condition)]

            paired = pd.concat(
                [ideal_values, condition_values], axis=1
            ).dropna()

            differences = (
                paired.iloc[:, 1].to_numpy()
                - paired.iloc[:, 0].to_numpy()
            )

            delta, low, high = paired_bootstrap(differences, rng)

            comparison_rows.append({
                "condition": condition,
                "metric": metric,
                "paired_poses": len(paired),
                "paired_delta_condition_minus_ideal": delta,
                "ci_lower_95": low,
                "ci_upper_95": high,
            })

        # Conditional errors: compare only poses where both conditions
        # returned at least one valid estimate.
        for metric in ["translation_error_mm", "axis_error_deg"]:
            ideal_values = wide[(metric, "ideal")]
            condition_values = wide[(metric, condition)]

            paired = pd.concat(
                [ideal_values, condition_values], axis=1
            ).dropna()

            differences = (
                paired.iloc[:, 1].to_numpy()
                - paired.iloc[:, 0].to_numpy()
            )

            delta, low, high = paired_bootstrap(differences, rng)

            comparison_rows.append({
                "condition": condition,
                "metric": metric,
                "paired_poses": len(paired),
                "paired_delta_condition_minus_ideal": delta,
                "ci_lower_95": low,
                "ci_upper_95": high,
            })

    comparisons = pd.DataFrame(comparison_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "statistical_summary.csv", index=False)
    comparisons.to_csv(OUT / "paired_bootstrap_comparisons.csv", index=False)
    per_pose.to_csv(OUT / "per_pose_aggregated_results.csv", index=False)

    payload = {
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": SEED,
        "independent_sampling_unit": "sample_id (pose)",
        "repeated_trials_aggregated_within_pose": True,
        "geometric_success_definition": (
            "translation and axis errors within explicit tolerances; "
            "failed estimates count as unsuccessful"
        ),
        "geometric_tolerances": {
            "strict": {"translation_mm": 5.0, "axis_deg": 5.0},
            "relaxed": {"translation_mm": 10.0, "axis_deg": 5.0},
        },
        "conditional_error_comparison": (
            "paired poses with valid estimates in both conditions"
        ),
        "summary": summary.replace({np.nan: None}).to_dict("records"),
        "paired_comparisons": (
            comparisons.replace({np.nan: None}).to_dict("records")
        ),
    }

    (OUT / "statistical_analysis.json").write_text(
        json.dumps(payload, indent=2)
    )

    print("\nCONDITION SUMMARY")
    print(summary.to_string(index=False))

    print("\nPAIRED BOOTSTRAP COMPARISONS")
    print(comparisons.to_string(index=False))

    print("\nSaved:")
    print(OUT / "statistical_summary.csv")
    print(OUT / "paired_bootstrap_comparisons.csv")
    print(OUT / "per_pose_aggregated_results.csv")
    print(OUT / "statistical_analysis.json")


if __name__ == "__main__":
    main()