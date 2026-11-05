import json
import h5py
import numpy as np
import open3d as o3d

from pathlib import Path

DATA = Path("outputs/randomized_pose_dataset")
CATEGORY = 4

# Representative samples, including the severe visibility failures.
SAMPLE_IDS = [0, 19, 126, 189, 198]

K = None
CV_TO_BLENDER = np.diag([1., -1., -1., 1.])

def load_points(sample):
    sid = int(sample["sample_id"])

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
    z = depth[v, u].astype(float)

    points_cv = np.column_stack([
        (u - K[0, 2]) * z / K[0, 0],
        (v - K[1, 2]) * z / K[1, 1],
        z,
        np.ones_like(z),
    ])

    T_cw = np.asarray(sample["camera_to_world_blender"])
    points_world = (T_cw @ CV_TO_BLENDER @ points_cv.T).T[:, :3]

    return points_world


def analyse(points):
    # Estimate the horizontal longitudinal direction from the point cloud.
    xy = points[:, :2]
    centred = xy - xy.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centred.T @ centred)
    axis = eigenvectors[:, -1]
    axis /= np.linalg.norm(axis)

    # Estimate local surface normals from the observed depth points.
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.025,
            max_nn=30,
        )
    )

    normals = np.asarray(cloud.normals)
    horizontal_normals = normals[:, :2]
    horizontal_norms = np.linalg.norm(horizontal_normals, axis=1)

    valid = horizontal_norms > 0.5
    horizontal_normals = (
        horizontal_normals[valid]
        / horizontal_norms[valid, None]
    )

    alignment = np.abs(horizontal_normals @ axis)

    return {
        "points": len(points),
        "valid_normals": len(alignment),
        "end_like_20deg_pct": 100 * np.mean(alignment > np.cos(np.deg2rad(20))),
        "end_like_30deg_pct": 100 * np.mean(alignment > np.cos(np.deg2rad(30))),
        "side_like_20deg_pct": 100 * np.mean(alignment < np.sin(np.deg2rad(20))),
        "median_alignment": float(np.median(alignment)),
    }


with open(DATA / "pose_labels.json") as f:
    metadata = json.load(f)

K = np.asarray(metadata["intrinsics"], dtype=float)
samples = {int(s["sample_id"]): s for s in metadata["samples"]}

print("End-face normal diagnostic")
print("Normals are estimated from depth points; no ground-truth pose is used.")
print()

for sid in SAMPLE_IDS:
    points = load_points(samples[sid])

    if len(points) < 100:
        print(f"Sample {sid}: insufficient points")
        continue

    result = analyse(points)

    print(
        f"Sample {sid:3d} | "
        f"N={result['points']:4d} | "
        f"valid normals={result['valid_normals']:4d} | "
        f"end-like ±20°={result['end_like_20deg_pct']:5.1f}% | "
        f"end-like ±30°={result['end_like_30deg_pct']:5.1f}% | "
        f"side-like ±20°={result['side_like_20deg_pct']:5.1f}% | "
        f"median alignment={result['median_alignment']:.3f}")