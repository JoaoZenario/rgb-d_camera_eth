import h5py
import numpy as np

from pathlib import Path
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "randomized_pose_dataset" / "0.hdf5"

SIGMA_D = 0.06
SCALES = [2, 4, 8, 16]
SEED = 20261106


def main():
    # Load the actual rendered depth frame.
    with h5py.File(INPUT, "r") as f:
        depth = np.asarray(f["depth"])

    # Match the validity definition in depth_noise.py.
    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    if not np.any(valid):
        raise RuntimeError("No valid depth pixels found.")

    print(f"Input: {INPUT}")
    print(f"Depth shape: {depth.shape}")
    print(f"Valid pixels: {valid.sum()}")
    print()
    print("Pre-quantisation disparity-noise statistics")
    print(f"Nominal standard deviation: {SIGMA_D:.5f} px")
    print()

    # Independent reference condition.
    rng_noise, _, _ = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(SEED).spawn(3)
    ]

    independent_noise = rng_noise.normal(
        0.0,
        SIGMA_D,
        size=valid.sum(),
    )

    print(
        f"{'Independent':<18}"
        f"mean={independent_noise.mean():+.6f} px  "
        f"std={independent_noise.std():.6f} px"
    )

    # Correlated conditions.
    # Use the same seed for each scale so that the underlying
    # full-image white-noise draw is identical across scales.
    for scale in SCALES:
        rng_noise, _, _ = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(SEED).spawn(3)
        ]

        white_noise = rng_noise.normal(size=depth.shape)

        correlated_field = gaussian_filter(
            white_noise,
            sigma=scale,
            mode="reflect",
        )

        valid_field = correlated_field[valid]

        field_std = valid_field.std()

        if not np.isfinite(field_std) or field_std <= 0:
            raise RuntimeError(
                f"Invalid field standard deviation at scale {scale}."
            )

        correlated_field = (
            correlated_field - valid_field.mean()
        ) / field_std

        disparity_noise = SIGMA_D * correlated_field[valid]

        print(
            f"{'Correlated ' + str(scale) + ' px':<18}"
            f"mean={disparity_noise.mean():+.6f} px  "
            f"std={disparity_noise.std():.6f} px"
        )


if __name__ == "__main__":
    main()