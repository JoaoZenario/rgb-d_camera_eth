import h5py
import numpy as np

from src.sensor.depth_noise import (
    apply_depth_noise,
    find_object_boundaries,
)

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"

TARGET_CATEGORY = 4
SEEDS = range(10, 20)

with h5py.File(INPUT_FILE, "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

valid = (
    np.isfinite(depth)
    & (depth > 0)
    & (depth < 1000)
)

target = (segmentation == TARGET_CATEGORY) & valid

edges = find_object_boundaries(
    segmentation,
    depth=depth,
)

target_count = np.count_nonzero(target)
target_boundary = target & edges

print("========== TARGET BOUNDARY AUDIT ==========")
print("Valid target pixels:", target_count)
print(
    "Target pixels in depth-discontinuity boundary mask:",
    np.count_nonzero(target_boundary)
)
print(
    "Target boundary fraction:",
    f"{100*np.count_nonzero(target_boundary)/target_count:.2f}%"
)

conditions = {
    "range_only": {
        "gaussian_noise": False,
        "range_missing": True,
        "boundary_failures": False,
    },
    "boundary_only": {
        "gaussian_noise": False,
        "range_missing": False,
        "boundary_failures": True,
    },
    "combined_dropout": {
        "gaussian_noise": False,
        "range_missing": True,
        "boundary_failures": True,
    },
}

print("\n========== DROPOUT BY MECHANISM ==========")

for name, settings in conditions.items():

    rates = []

    for seed in SEEDS:
        _, invalid, _ = apply_depth_noise(
            depth,
            segmentation,
            seed=seed,
            **settings
        )

        target_invalid = np.count_nonzero(
            invalid & target
        )

        boundary_invalid = np.count_nonzero(
            invalid & target & edges
        )

        rate = target_invalid / target_count * 100
        boundary_rate = boundary_invalid / target_count * 100

        rates.append(rate)

        if name == "boundary_only":
            assert target_invalid == boundary_invalid, (
                "Boundary-only dropout occurred outside the boundary mask."
            )
            print(
                f"Seed {seed}: "
                f"target invalid={target_invalid}, "
                f"boundary invalid={boundary_invalid}, "
                f"boundary dropout={boundary_rate:.2f}%"
            )

    print(
        f"{name:18s} "
        f"mean={np.mean(rates):6.2f}% "
        f"SD={np.std(rates, ddof=1):5.2f}% "
        f"range={np.min(rates):.2f}–{np.max(rates):.2f}%"
    )

print("\n========== TARGET RANGE ==========")
target_depth = depth[target]

print(f"Minimum: {target_depth.min():.3f} m")
print(f"Maximum: {target_depth.max():.3f} m")
print(f"Mean:    {target_depth.mean():.3f} m")