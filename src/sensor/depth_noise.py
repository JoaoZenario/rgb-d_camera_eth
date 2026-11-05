import numpy as np

from scipy.ndimage import gaussian_filter

# Nominal D455-inspired camera parameters.
# These are simulation inputs, not physical calibration results.
DEFAULT_FOCAL_LENGTH_PX = 343.1579872078984
DEFAULT_BASELINE_M = 0.095

# Explicit phenomenological assumptions, not measured D455 parameters.
DEFAULT_DISPARITY_SIGMA_PX = 0.06
DEFAULT_DISPARITY_QUANTIZATION_PX = 1.0 / 32.0
DEFAULT_CORRELATION_SIGMA_PX = 8.0

DEFAULT_BOUNDARY_PROBABILITY = 0.05
DEFAULT_DEPTH_DISCONTINUITY_M = 0.05


def find_object_boundaries(
    segmentation,
    depth=None,
    min_depth_jump_m=DEFAULT_DEPTH_DISCONTINUITY_M,
):
    """
    Identify thin boundaries at sufficiently large depth discontinuities.

    If depth is provided, semantic transitions are retained only where
    adjacent valid depth measurements differ by at least min_depth_jump_m.

    Both pixels on either side of a qualifying transition are marked.
    No morphological expansion is applied.
    """

    height, width = segmentation.shape
    edge_mask = np.zeros((height, width), dtype=bool)

    valid = None

    if depth is not None:
        if depth.shape != segmentation.shape:
            raise ValueError(
                "Depth and segmentation must have the same shape."
            )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & (depth < 1000)
        )

    # Horizontal transitions: mark both adjacent pixels.
    horizontal_transition = (
        segmentation[:, 1:] != segmentation[:, :-1]
    )

    if depth is not None:
        horizontal_valid = (
            valid[:, 1:] & valid[:, :-1]
        )

        horizontal_jump = np.abs(
            depth[:, 1:] - depth[:, :-1]
        )

        horizontal_transition &= (
            horizontal_valid
            & (horizontal_jump >= min_depth_jump_m)
        )

    edge_mask[:, 1:] |= horizontal_transition
    edge_mask[:, :-1] |= horizontal_transition

    # Vertical transitions: mark both adjacent pixels.
    vertical_transition = (
        segmentation[1:, :] != segmentation[:-1, :]
    )

    if depth is not None:
        vertical_valid = (
            valid[1:, :] & valid[:-1, :]
        )

        vertical_jump = np.abs(
            depth[1:, :] - depth[:-1, :]
        )

        vertical_transition &= (
            vertical_valid
            & (vertical_jump >= min_depth_jump_m)
        )

    edge_mask[1:, :] |= vertical_transition
    edge_mask[:-1, :] |= vertical_transition

    return edge_mask


def apply_depth_noise(
    depth,
    segmentation,
    seed=42,
    gaussian_noise=True,
    range_missing=True,
    boundary_failures=True,
    focal_length_px=DEFAULT_FOCAL_LENGTH_PX,
    baseline_m=DEFAULT_BASELINE_M,
    disparity_sigma_px=DEFAULT_DISPARITY_SIGMA_PX,
    disparity_quantization_px=DEFAULT_DISPARITY_QUANTIZATION_PX,
    disparity_noise_mode="independent",
    correlation_sigma_px=DEFAULT_CORRELATION_SIGMA_PX,
    boundary_probability=DEFAULT_BOUNDARY_PROBABILITY,
    depth_discontinuity_threshold_m=DEFAULT_DEPTH_DISCONTINUITY_M,
):
    """
    Apply configurable phenomenological stereo-depth corruption.

    Parameters
    ----------
    depth : ndarray
        Ideal axial depth map in metres.
    segmentation : ndarray
        Semantic segmentation map.
    seed : int
        Random seed for reproducibility.
    gaussian_noise : bool
        Enable Gaussian disparity noise and disparity quantisation.
    range_missing : bool
        Enable range-dependent invalid measurements.
    boundary_failures : bool
        Enable additional invalid measurements near large depth jumps.
    focal_length_px : float
        Horizontal focal length in pixels.
    baseline_m : float
        Stereo baseline in metres.
    disparity_sigma_px : float
        Assumed standard deviation of disparity noise in pixels.
    disparity_quantization_px : float
        Assumed disparity quantisation interval in pixels.
    disparity_noise_mode : str
        "independent" preserves the original per-pixel Gaussian model.
        "correlated" uses a spatially smoothed Gaussian random field.
    correlation_sigma_px : float
        Gaussian smoothing standard deviation in pixels for correlated
        disparity noise. This is a phenomenological assumption.
    boundary_probability : float
        Assumed probability of invalid depth at qualifying boundaries.
    depth_discontinuity_threshold_m : float
        Minimum adjacent depth difference for a boundary to qualify.

    Returns
    -------
    corrupted_depth : ndarray
        Corrupted depth map. Invalid measurements are NaN.
    invalid_mask : ndarray
        Pixels valid in the original map but invalidated by corruption.
    edge_mask : ndarray
        Thin mask of semantic transitions with sufficiently large
        depth discontinuities.
    """

    if depth.shape != segmentation.shape:
        raise ValueError(
            "Depth and segmentation must have the same shape."
        )

    if focal_length_px <= 0 or baseline_m <= 0:
        raise ValueError(
            "Focal length and baseline must be positive."
        )

    if disparity_sigma_px < 0:
        raise ValueError(
            "Disparity noise standard deviation cannot be negative."
        )

    if disparity_quantization_px <= 0:
        raise ValueError(
            "Disparity quantisation interval must be positive."
        )

    if disparity_noise_mode not in ("independent", "correlated"):
        raise ValueError(
            "disparity_noise_mode must be 'independent' or 'correlated'."
        )

    if correlation_sigma_px <= 0:
        raise ValueError(
            "Correlation sigma must be positive."
        )

    if not 0.0 <= boundary_probability <= 1.0:
        raise ValueError(
            "Boundary probability must be between 0 and 1."
        )

    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    edge_mask = find_object_boundaries(
        segmentation,
        depth=depth,
        min_depth_jump_m=depth_discontinuity_threshold_m,
    )

    # Independent random streams ensure reproducible ablations.
    rng_noise, rng_range, rng_boundary = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(3)
    ]

    corrupted_depth = np.full(
        depth.shape,
        np.nan,
        dtype=np.float32,
    )

    z = depth[valid].astype(np.float64)

    # Start with ideal depth values.
    noisy_z = z.copy()

    # 1. Disparity-domain measurement noise.
    if gaussian_noise:
        disparity = (
            focal_length_px * baseline_m / z
        )

        if disparity_noise_mode == "independent":
            # Original model: independent Gaussian noise at each pixel.
            disparity_noise = rng_noise.normal(
                0.0,
                disparity_sigma_px,
                size=disparity.shape,
            )

        else:
            # Correlated model: smooth a white-noise field spatially.
            # Normalisation preserves the marginal standard deviation
            # over valid pixels before disparity quantisation.
            white_noise = rng_noise.normal(size=depth.shape)

            correlated_field = gaussian_filter(
                white_noise,
                sigma=correlation_sigma_px,
                mode="reflect",
            )

            valid_field = correlated_field[valid]
            field_std = valid_field.std()

            if not np.isfinite(field_std) or field_std <= 0:
                raise RuntimeError(
                    "Could not normalise correlated disparity field."
                )

            correlated_field = (
                correlated_field - valid_field.mean()
            ) / field_std

            disparity_noise = (
                disparity_sigma_px * correlated_field[valid]
            )

        noisy_disparity = disparity + disparity_noise

        # Quantise disparity using the assumed subpixel interval.
        noisy_disparity = (
            np.round(
                noisy_disparity / disparity_quantization_px
            )
            * disparity_quantization_px
        )

        # Non-positive disparity cannot produce a valid positive depth.
        valid_disparity = noisy_disparity > 0

        noisy_z[valid_disparity] = (
            focal_length_px
            * baseline_m
            / noisy_disparity[valid_disparity]
        )

        noisy_z[~valid_disparity] = np.nan

    # 2. Range-dependent invalid measurements.
    invalid = np.zeros(depth.shape, dtype=bool)

    if range_missing:
        p_range = (
            0.002
            + 0.03 * (z / 6.0) ** 2
        )

        p_range = np.clip(p_range, 0.0, 1.0)

        invalid[valid] |= (
            rng_range.random(z.shape) < p_range
        )

    # 3. Additional failures at large depth discontinuities.
    if boundary_failures and boundary_probability > 0:
        boundary_valid = valid & edge_mask

        n_boundary = np.count_nonzero(boundary_valid)

        if n_boundary:
            invalid[boundary_valid] |= (
                rng_boundary.random(n_boundary)
                < boundary_probability
            )

    # Invalid disparity values are also invalid measurements.
    if gaussian_noise:
        invalid_values = ~np.isfinite(noisy_z)

        invalid[valid] |= invalid_values

    # Assign noisy depths only to surviving valid pixels.
    keep = valid & ~invalid

    corrupted_depth[keep] = noisy_z[~invalid[valid]].astype(
        np.float32
    )

    invalid_mask = valid & invalid

    return corrupted_depth, invalid_mask, edge_mask