import numpy as np
import pandas as pd

INPUT = "outputs/randomized_pose_ablation/trial_results.csv"
OUTPUT = "results/randomized_pose_ablation/cross_axis_grasp_sensitivity.csv"

df = pd.read_csv(INPUT)

conditions = [
    "ideal",
    "disparity_only",
    "dropout_only",
    "full_corruption",
]

thresholds = [5, 10, 15, 20, 25]
axis_threshold = 5.0
B = 10000
rng = np.random.default_rng(20261105)

rows = []

for threshold in thresholds:
    for condition in conditions:
        subset = df[df["condition"] == condition].copy()

        subset["success"] = (
            subset["estimator_valid"].astype(bool)
            & (subset["cross_axis_error_mm"] <= threshold)
            & (subset["long_axis_error_deg"] <= axis_threshold)
        )

        scores = (
            subset.groupby("sample_id")["success"]
            .mean()
            .sort_index()
        )

        rows.append({
            "cross_axis_threshold_mm": threshold,
            "axis_threshold_deg": axis_threshold,
            "condition": condition,
            "num_poses": len(scores),
            "success_rate": scores.mean(),
        })

    # Paired pose-level comparisons against ideal.
    scores_by_condition = {}

    for condition in conditions:
        subset = df[df["condition"] == condition].copy()

        subset["success"] = (
            subset["estimator_valid"].astype(bool)
            & (subset["cross_axis_error_mm"] <= threshold)
            & (subset["long_axis_error_deg"] <= axis_threshold)
        )

        scores_by_condition[condition] = (
            subset.groupby("sample_id")["success"].mean()
        )

    ideal = scores_by_condition["ideal"]

    for condition in conditions[1:]:
        paired = pd.concat(
            [ideal.rename("ideal"),
             scores_by_condition[condition].rename("other")],
            axis=1,
            join="inner",
        ).dropna()

        delta = (
            paired["other"] - paired["ideal"]
        ).to_numpy()

        boot = np.empty(B)

        for b in range(B):
            sample = rng.choice(delta, size=len(delta), replace=True)
            boot[b] = sample.mean()

        rows.append({
            "cross_axis_threshold_mm": threshold,
            "axis_threshold_deg": axis_threshold,
            "condition": condition,
            "num_poses": len(paired),
            "success_rate": np.nan,
            "paired_delta_vs_ideal": delta.mean(),
            "ci_lower_95": np.percentile(boot, 2.5),
            "ci_upper_95": np.percentile(boot, 97.5),
        })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)

print(out.to_string(index=False))
print(f"\nSaved: {OUTPUT}")