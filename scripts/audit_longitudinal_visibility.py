import json
import h5py
import numpy as np

from pathlib import Path

DATA = Path("outputs/randomized_pose_dataset")
CATEGORY = 4
HALF_LENGTH = 0.65
HALF_WIDTH = 0.04
HALF_HEIGHT = 0.04

with open(DATA / "pose_labels.json") as f:
    metadata = json.load(f)

records = []

for sample in metadata["samples"]:
    sid = int(sample["sample_id"])
    T = np.asarray(sample["target_to_world"], dtype=float)

    with h5py.File(DATA / f"{sid}.hdf5", "r") as f:
        depth = np.squeeze(f["depth"][:])
        seg = np.squeeze(f["category_id_segmaps"][:])

    mask = (
        (seg == CATEGORY)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 1000)
    )

    v, u = np.where(mask)
    if len(u) < 100:
        continue

    z = depth[v, u].astype(float)
    K = np.asarray(metadata["intrinsics"], dtype=float)

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    CV_TO_BLENDER = np.diag([1., -1., -1., 1.])
    T_cw = np.asarray(sample["camera_to_world_blender"], dtype=float)

    points_world = (T_cw @ CV_TO_BLENDER @ points_cv.T).T
    points_local = (np.linalg.inv(T) @ points_world.T).T[:, :3]

    x = points_local[:, 0]
    y = points_local[:, 1]
    zz = points_local[:, 2]

    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])

    # End-face coverage: points close to either x end.
    end_left = np.mean(x < -HALF_LENGTH + 0.015)
    end_right = np.mean(x > HALF_LENGTH - 0.015)

    # Side-face coverage: points close to either y side.
    side = np.mean(np.abs(np.abs(y) - HALF_WIDTH) < 0.005)

    records.append({
        "sample_id": sid,
        "points": len(x),
        "visible_length_pct": 100 * (xhi - xlo) / (2 * HALF_LENGTH),
        "visible_midpoint_offset_mm": 500 * (xhi + xlo),
        "left_end_fraction_pct": 100 * end_left,
        "right_end_fraction_pct": 100 * end_right,
        "side_surface_fraction_pct": 100 * side,
    })

def stats(key):
    a = np.array([r[key] for r in records])
    return (
        np.median(a),
        np.percentile(a, 10),
        np.percentile(a, 90),
        np.min(a),
        np.max(a),
    )

print(f"Analysed samples: {len(records)}")
for key in [
    "visible_length_pct",
    "visible_midpoint_offset_mm",
    "left_end_fraction_pct",
    "right_end_fraction_pct",
    "side_surface_fraction_pct",
]:
    med, p10, p90, lo, hi = stats(key)
    print(
        f"{key}: median={med:.2f}, "
        f"P10={p10:.2f}, P90={p90:.2f}, "
        f"min={lo:.2f}, max={hi:.2f}"
    )

print("\nLeast-visible samples:")
for r in sorted(records, key=lambda x: x["visible_length_pct"])[:10]:
    print(r)

out = DATA / "longitudinal_visibility_audit.json"
out.write_text(json.dumps(records, indent=2))
print("\nSaved:", out)
