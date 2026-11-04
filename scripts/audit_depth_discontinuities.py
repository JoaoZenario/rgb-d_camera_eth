import h5py
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

INPUT_FILE = "outputs/manufacturing_scene/0.hdf5"
TARGET_CATEGORY = 4

# Explore several thresholds rather than assuming one is correct.
DEPTH_THRESHOLDS_M = (0.01, 0.025, 0.05)


def valid_depth(depth):
    return (
        np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )


def collect_transitions(segmentation, depth, valid):
    """
    Collect horizontal and vertical neighbouring pixel pairs
    where semantic labels differ and both depth values are valid.
    """
    transitions = []

    # Horizontal pairs: left pixel and right pixel.
    label_change = (
        segmentation[:, :-1] != segmentation[:, 1:]
    )
    pair_valid = valid[:, :-1] & valid[:, 1:]
    rows, cols = np.where(label_change & pair_valid)

    for y, x in zip(rows, cols):
        transitions.append((
            (y, x),
            (y, x + 1),
            abs(float(depth[y, x]) - float(depth[y, x + 1])),
        ))

    # Vertical pairs: upper pixel and lower pixel.
    label_change = (
        segmentation[:-1, :] != segmentation[1:, :]
    )
    pair_valid = valid[:-1, :] & valid[1:, :]
    rows, cols = np.where(label_change & pair_valid)

    for y, x in zip(rows, cols):
        transitions.append((
            (y, x),
            (y + 1, x),
            abs(float(depth[y, x]) - float(depth[y + 1, x])),
        ))

    return transitions


def target_endpoint_mask(shape, transitions, threshold, target):
    """
    Mark target pixels participating in semantic transitions
    whose depth jump exceeds the threshold.
    """
    mask = np.zeros(shape, dtype=bool)

    for p1, p2, jump in transitions:
        if jump < threshold:
            continue

        if target[p1]:
            mask[p1] = True

        if target[p2]:
            mask[p2] = True

    return mask


with h5py.File(INPUT_FILE, "r") as f:
    depth = np.squeeze(f["depth"][...]).astype(np.float32)
    segmentation = np.squeeze(
        f["category_id_segmaps"][...]
    )

if depth.shape != segmentation.shape:
    raise ValueError("Depth and segmentation shapes do not match.")

valid = valid_depth(depth)
target = (segmentation == TARGET_CATEGORY) & valid
target_count = np.count_nonzero(target)

if target_count == 0:
    raise RuntimeError("No valid target pixels found.")

transitions = collect_transitions(
    segmentation,
    depth,
    valid,
)

jumps = np.array(
    [item[2] for item in transitions],
    dtype=np.float32,
)

print("\n========== DEPTH DISCONTINUITY AUDIT ==========")
print("Valid target pixels:", target_count)
print("Valid semantic transition pairs:", len(transitions))

pair_stats = defaultdict(list)
target_pair_stats = defaultdict(list)

for p1, p2, jump in transitions:
    label1 = int(segmentation[p1])
    label2 = int(segmentation[p2])

    pair = tuple(sorted((label1, label2)))
    pair_stats[pair].append(jump)

    if target[p1] or target[p2]:
        target_pair_stats[pair].append(jump)

print("\n========== SEMANTIC CATEGORY PAIRS ==========")

for pair, values in sorted(
    pair_stats.items(),
    key=lambda item: len(item[1]),
    reverse=True,
):
    values = np.asarray(values)

    print(
        f"{pair[0]} <-> {pair[1]}: "
        f"N={len(values):4d}, "
        f"median={np.median(values)*1000:7.2f} mm, "
        f"P90={np.percentile(values, 90)*1000:7.2f} mm, "
        f">50mm={100*np.mean(values >= 0.05):6.2f}%"
    )

print("\n========== TARGET-ADJACENT CATEGORY PAIRS ==========")

print("\n========== TARGET DEPTH ORDERING ==========")

ordering_stats = {}

for p1, p2, jump in transitions:
    label1 = int(segmentation[p1])
    label2 = int(segmentation[p2])

    if label1 == TARGET_CATEGORY:
        target_pixel = p1
        neighbour_pixel = p2
        neighbour_label = label2
    elif label2 == TARGET_CATEGORY:
        target_pixel = p2
        neighbour_pixel = p1
        neighbour_label = label1
    else:
        continue

    z_target = float(depth[target_pixel])
    z_neighbour = float(depth[neighbour_pixel])

    if z_target < z_neighbour:
        ordering = "target_nearer"
    elif z_target > z_neighbour:
        ordering = "target_farther"
    else:
        ordering = "equal"

    key = (neighbour_label, ordering)
    ordering_stats.setdefault(key, []).append(jump)

for (neighbour_label, ordering), values in sorted(
    ordering_stats.items()
):
    values = np.asarray(values)

    print(
        f"Target <-> category {neighbour_label}, "
        f"{ordering:14s}: "
        f"N={len(values):4d}, "
        f"median jump={np.median(values)*1000:7.2f} mm, "
        f">50mm={100*np.mean(values >= 0.05):6.2f}%"
    )

for pair, values in sorted(
    target_pair_stats.items(),
    key=lambda item: len(item[1]),
    reverse=True,
):
    values = np.asarray(values)

    print(
        f"{pair[0]} <-> {pair[1]}: "
        f"N={len(values):4d}, "
        f"median={np.median(values)*1000:7.2f} mm, "
        f"P90={np.percentile(values, 90)*1000:7.2f} mm, "
        f">50mm={100*np.mean(values >= 0.05):6.2f}%"
    )

if len(jumps) == 0:
    raise RuntimeError("No valid semantic transitions found.")

print("\nDepth jump across semantic transitions:")
print(f"  Median: {np.median(jumps) * 1000:.2f} mm")
print(f"  P75:    {np.percentile(jumps, 75) * 1000:.2f} mm")
print(f"  P90:    {np.percentile(jumps, 90) * 1000:.2f} mm")
print(f"  P95:    {np.percentile(jumps, 95) * 1000:.2f} mm")
print(f"  Maximum:{np.max(jumps) * 1000:.2f} mm")

print("\nTarget pixels participating in transitions:")
for threshold in DEPTH_THRESHOLDS_M:
    mask = target_endpoint_mask(
        depth.shape,
        transitions,
        threshold,
        target,
    )

    count = np.count_nonzero(mask)
    fraction = 100.0 * count / target_count

    print(
        f"  Depth jump >= {threshold * 1000:5.0f} mm: "
        f"{count:4d} pixels ({fraction:6.2f}%)"
    )

# Build an image showing target pixels adjacent to large depth jumps.
fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9),
)

axes = axes.ravel()

axes[0].imshow(depth, cmap="viridis")
axes[0].set_title("Ideal depth map (m)")

axes[1].imshow(segmentation, cmap="tab20", interpolation="nearest")
axes[1].set_title("Semantic segmentation")

for ax, threshold in zip(
    axes[2:],
    (0.01, 0.05),
):
    mask = target_endpoint_mask(
        depth.shape,
        transitions,
        threshold,
        target,
    )

    ax.imshow(target, cmap="gray", alpha=0.8)
    ax.imshow(
        np.ma.masked_where(~mask, mask),
        cmap="Reds",
        alpha=1.0,
        interpolation="nearest",
    )

    ax.set_title(
        f"Target boundary pixels: depth jump >= "
        f"{threshold * 1000:.0f} mm"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = "report/figures/depth_discontinuity_audit.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close()

# Also save the distribution of depth jumps.
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(jumps * 1000, bins=60)
ax.set_xlabel("Absolute depth difference across transition (mm)")
ax.set_ylabel("Number of neighbouring pixel pairs")
ax.set_title("Depth differences at semantic boundaries")
ax.grid(True, alpha=0.3)

hist_path = "report/figures/depth_jump_distribution.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=180, bbox_inches="tight")
plt.close()

print("\nSaved figures:")
print(" ", output_path)
print(" ", hist_path)