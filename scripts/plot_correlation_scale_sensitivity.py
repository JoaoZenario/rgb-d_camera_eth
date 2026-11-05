import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "correlation_scale_sweep" / "summary.json"
OUTPUT = ROOT / "report" / "figures" / "correlation_scale_sensitivity.png"

SCALES = [2, 4, 8, 16]


def get_comparison(summary, scale, metric):
    """Return paired mean difference and its 95% bootstrap CI."""
    condition = f"correlated_{scale}px"

    comparison = summary[
        "paired_comparisons_vs_independent"
    ][condition][metric]

    mean = comparison["mean_delta_candidate_minus_independent"]
    ci = comparison["bootstrap_95_ci"]

    lower = ci["lower_95"]
    upper = ci["upper_95"]

    return mean, lower, upper


def main():
    with INPUT.open("r") as f:
        summary = json.load(f)

    cross = [
        get_comparison(summary, scale, "cross")
        for scale in SCALES
    ]

    axis = [
        get_comparison(summary, scale, "axis")
        for scale in SCALES
    ]

    cross_mean = np.array([v[0] for v in cross])
    cross_lower = np.array([v[1] for v in cross])
    cross_upper = np.array([v[2] for v in cross])

    axis_mean = np.array([v[0] for v in axis])
    axis_lower = np.array([v[1] for v in axis])
    axis_upper = np.array([v[2] for v in axis])

    cross_errors = np.vstack([
        cross_mean - cross_lower,
        cross_upper - cross_mean,
    ])

    axis_errors = np.vstack([
        axis_mean - axis_lower,
        axis_upper - axis_mean,
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # Cross-axis translation error.
    axes[0].errorbar(
        SCALES,
        cross_mean,
        yerr=cross_errors,
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Gaussian correlation scale (pixels)")
    axes[0].set_ylabel(
        "Change in mean cross-axis error (mm)"
    )
    axes[0].set_title("Cross-axis translation error")
    axes[0].set_xticks(SCALES)
    axes[0].grid(True, alpha=0.3)

    # Dominant-axis orientation error.
    axes[1].errorbar(
        SCALES,
        axis_mean,
        yerr=axis_errors,
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Gaussian correlation scale (pixels)")
    axes[1].set_ylabel(
        "Change in mean dominant-axis error (degrees)"
    )
    axes[1].set_title("Dominant-axis orientation error")
    axes[1].set_xticks(SCALES)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT}")


if __name__ == "__main__":
    main()