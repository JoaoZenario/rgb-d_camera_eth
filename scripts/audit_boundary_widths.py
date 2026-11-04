import h5py
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
TARGET_CATEGORY = 4

def dilate(mask, radius):
    """Square/Chebyshev dilation by the specified pixel radius."""
    if radius == 0:
        return mask.copy()

    h, w = mask.shape
    padded = np.pad(
        mask,
        radius,
        mode="constant",
        constant_values=False,
    )

    result = np.zeros_like(mask, dtype=bool)

    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[
                dy:dy + h,
                dx:dx + w,
            ]

    return result


def one_sided_edges(segmentation):
    """Reproduce the original edge definition."""
    edges = np.zeros(segmentation.shape, dtype=bool)

    edges[:, 1:] |= (
        segmentation[:, 1:] != segmentation[:, :-1]
    )
    edges[1:, :] |= (
        segmentation[1:, :] != segmentation[:-1, :]
    )

    return edges


def symmetric_edges(segmentation):
    """Mark both pixels on either side of horizontal/vertical transitions."""
    edges = np.zeros(segmentation.shape, dtype=bool)

    horizontal = segmentation[:, 1:] != segmentation[:, :-1]
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal

    vertical = segmentation[1:, :] != segmentation[:-1, :]
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical

    return edges


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
target_count = np.count_nonzero(target)

if target_count == 0:
    raise RuntimeError("No valid target pixels found.")

# Compare the original one-sided edge definition with a symmetric one.
original_edges = one_sided_edges(segmentation)
symmetric = symmetric_edges(segmentation)

print("\n========== BOUNDARY WIDTH AUDIT ==========")
print("Valid target pixels:", target_count)
print("\nOriginal one-sided edge definition:")

for radius in (0, 1, 2):
    mask = dilate(original_edges, radius)
    covered = np.count_nonzero(target & mask)
    fraction = 100.0 * covered / target_count

    print(
        f"  Expansion radius {radius}: "
        f"{covered:4d} pixels ({fraction:6.2f}%)"
    )

print("\nSymmetric edge definition:")

for radius in (0, 1, 2):
    mask = dilate(symmetric, radius)
    covered = np.count_nonzero(target & mask)
    fraction = 100.0 * covered / target_count

    print(
        f"  Expansion radius {radius}: "
        f"{covered:4d} pixels ({fraction:6.2f}%)"
    )

# Compare the actual original mask with the symmetric radius-1 mask.
original_radius2 = dilate(original_edges, 2)
symmetric_radius1 = dilate(symmetric, 1)

print("\nOriginal radius-2 versus symmetric radius-1:")
for name, mask in [
    ("Original radius-2", original_radius2),
    ("Symmetric radius-1", symmetric_radius1),
]:
    covered = np.count_nonzero(target & mask)
    print(
        f"  {name}: "
        f"{100.0 * covered / target_count:.2f}% of target"
    )

# Visualise the target and the boundary regions.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

views = [
    ("Target mask", target, False),
    ("Original edges, no expansion", original_edges, True),
    ("Original edges, radius 2", original_radius2, True),
    ("Symmetric edges, no expansion", symmetric, True),
    ("Symmetric edges, radius 1", symmetric_radius1, True),
    ("Symmetric edges, radius 2", dilate(symmetric, 2), True),
]

for ax, (title, mask, show_target) in zip(axes.flat, views):
    ax.imshow(segmentation, cmap="tab20", interpolation="nearest")

    if show_target:
        # Outline the valid target pixels.
        ax.contour(
            target.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.5,
        )

        # Show the boundary mask in red, restricted to valid depth pixels.
        overlay = np.ma.masked_where(
            ~(mask & valid),
            mask & valid,
        )
        ax.imshow(
            overlay,
            cmap="Reds",
            alpha=0.8,
            interpolation="nearest",
        )
    else:
        ax.imshow(
            target,
            cmap="Reds",
            alpha=0.8,
            interpolation="nearest",
        )

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

output_path = "report/figures/boundary_width_audit.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nSaved visual audit to: {output_path}")