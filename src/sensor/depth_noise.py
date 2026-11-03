import numpy as np

def find_object_boundaries(segmentation):
    """Find and expand semantic boundaries by two pixels."""
    edge_mask = np.zeros(segmentation.shape, dtype=bool)

    edge_mask[:, 1:] |= (
        segmentation[:, 1:] != segmentation[:, :-1]
    )
    edge_mask[1:, :] |= (
        segmentation[1:, :] != segmentation[:-1, :]
    )

    # Expand the boundary region by two pixels.
    expanded = edge_mask.copy()

    for _ in range(2):
        padded = np.pad(expanded, 1, mode="constant")
        expanded = np.zeros_like(expanded)

        for dy in range(3):
            for dx in range(3):
                expanded |= padded[
                    dy:dy + segmentation.shape[0],
                    dx:dx + segmentation.shape[1]
                ]

    return expanded


def apply_depth_noise(
    depth,
    segmentation,
    seed=42,
    gaussian_noise=True,
    range_missing=True,
    boundary_failures=True,
):
    """
    Apply configurable phenomenological depth corruption.

    Parameters
    ----------
    depth : ndarray
        Ideal depth map in metres.
    segmentation : ndarray
        Semantic segmentation map.
    seed : int
        Random seed for reproducibility.
    gaussian_noise : bool
        Enable range-dependent Gaussian measurement noise.
    range_missing : bool
        Enable range-dependent invalid measurements.
    boundary_failures : bool
        Enable additional invalid measurements near object boundaries.

    Returns
    -------
    corrupted_depth : ndarray
        Corrupted depth map. Invalid measurements are NaN.
    invalid_mask : ndarray
        Pixels that were valid in the original map but invalidated.
    edge_mask : ndarray
        Expanded object-boundary mask.
    """

    if depth.shape != segmentation.shape:
        raise ValueError(
            "Depth and segmentation must have the same shape."
        )

    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    edge_mask = find_object_boundaries(segmentation)

    # Independent random streams make each effect reproducible.
    rng_noise, rng_range, rng_boundary = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(3)
    ]

    corrupted_depth = np.full(depth.shape, np.nan, dtype=np.float32)

    z = depth[valid].astype(np.float32)
    noisy_z = z.copy()

    # 1. Range-dependent Gaussian noise.
    if gaussian_noise:
        sigma = 0.001 + 0.0015 * z**2
        noisy_z += rng_noise.normal(0.0, sigma).astype(np.float32)

    # 2. Range-dependent missing measurements.
    invalid = np.zeros(depth.shape, dtype=bool)

    if range_missing:
        p_range = 0.002 + 0.03 * (z / 6.0)**2
        invalid[valid] |= (
            rng_range.random(z.shape) < p_range
        )

    # 3. Additional missing measurements near boundaries.
    if boundary_failures:
        p_boundary = 0.15
        boundary_valid = valid & edge_mask

        invalid[boundary_valid] |= (
            rng_boundary.random(np.count_nonzero(boundary_valid))
            < p_boundary
        )

    # Store noisy values only where measurements remain valid.
    keep = valid & ~invalid
    corrupted_depth[keep] = noisy_z[~invalid[valid]]

    invalid_mask = valid & invalid

    return corrupted_depth, invalid_mask, edge_mask