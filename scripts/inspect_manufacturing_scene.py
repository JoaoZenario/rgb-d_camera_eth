
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Load rendered data
file_path = "outputs/manufacturing_scene/0.hdf5"

with h5py.File(file_path, "r") as f:

    print("Available datasets:")
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)

    rgb = np.array(f["colors"])
    depth = np.array(f["depth"])
    segmentation = np.array(f["category_id_segmaps"])

# Remove batch dimension if present
if rgb.ndim == 4:
    rgb = rgb[0]

if depth.ndim == 3:
    depth = depth[0]

if segmentation.ndim == 3:
    segmentation = segmentation[0]

# Convert depth into a displayable image
valid = np.isfinite(depth) & (depth < 1000)

depth_display = np.zeros_like(depth, dtype=np.float32)

if np.any(valid):
    depth_display[valid] = depth[valid]

# Display outputs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb)
axes[0].set_title("RGB image")

axes[1].imshow(depth_display, cmap="viridis")
axes[1].set_title("Ideal depth")

axes[2].imshow(segmentation, cmap="tab10", vmin=0, vmax=3)
axes[2].set_title("Semantic segmentation")

for ax in axes:
    ax.axis("off")

os.makedirs("report/figures", exist_ok=True)

plt.tight_layout()

plt.savefig(
    "report/figures/manufacturing_scene_overview.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to report/figures/manufacturing_scene_overview.png")
