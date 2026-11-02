
import numpy as np


def find_object_boundaries(segmentation):
    """
    Find pixels close to semantic class boundaries.

    The boundary is expanded by two pixels to model the
    increased likelihood of stereo matching failures
    around object edges.
    """
    boundaries = np.zeros(segmentation.shape, dtype=bool)

    # Find horizontal class transitions
    diff = segmentation[:, 1:] != segmentation[:, :-1]
    boundaries[:, 1:] |= diff
    boundaries[:, :-1] |= diff

    # Find vertical class transitions
    diff = segmentation[1:, :] != segmentation[:-1, :]
    boundaries[1:, :] |= diff
    boundaries[:-1, :] |= diff

    # Expand the boundary region by two pixels
    expanded = boundaries.copy()

    for _ in range(2):
        padded = np.pad(
            expanded,
            pad_width=1,
            mode="constant",
            constant_values=False
        )

        expanded = np.zeros_like(expanded)

        for dy in range(3):
            for dx in range(3):
                expanded |= padded[
                    dy:dy + segmentation.shape[0],
                    dx:dx + segmentation.shape[1]
                ]

    return expanded


def apply_depth_noise(depth, segmentation, seed=42):
    """
    Apply a phenomenological stereo depth sensor model.

    Parameters
    ----------
    depth : np.ndarray
        Ideal depth map in metres.
    segmentation : np.ndarray
        Semantic class map.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    corrupted_depth : np.ndarray
        Simulated sensor depth in metres. Invalid pixels
        are represented by NaN.
    invalid_mask : np.ndarray
        Boolean mask of invalid depth measurements.
    edge_mask : np.ndarray
        Boolean mask of pixels near semantic boundaries.
    """

    rng = np.random.default_rng(seed)

    depth = np.asarray(depth, dtype=np.float32)
    segmentation = np.asarray(segmentation)

    if depth.shape != segmentation.shape:
        raise ValueError(
            "Depth and segmentation maps must have the same shape."
        )

    # Identify valid rendered depth measurements
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1000)

    corrupted_depth = np.full(depth.shape, np.nan, dtype=np.float32)

    # Find object boundaries
    edge_mask = find_object_boundaries(segmentation)

    z = depth[valid]

    # 1. Range-dependent Gaussian depth noise
    # sigma(z) = a + b * z^2
    sigma = 0.001 + 0.0015 * z**2

    noisy_depth = z + rng.normal(0.0, sigma)

    # 2. Range-dependent probability of invalid measurements
    range_probability = 0.002 + 0.03 * (z / 6.0)**2

    # 3. Additional invalid measurements near object boundaries
    probabilities = range_probability.copy()
    probabilities += 0.15 * edge_mask[valid]

    probabilities = np.clip(probabilities, 0.0, 1.0)

    random_values = rng.random(z.shape)
    invalid = random_values < probabilities

    # Store valid noisy measurements
    noisy_depth[invalid] = np.nan
    corrupted_depth[valid] = noisy_depth

    invalid_mask = ~np.isfinite(corrupted_depth)

    return corrupted_depth, invalid_mask, edge_mask
