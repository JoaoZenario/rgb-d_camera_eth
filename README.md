# Synthetic RGB-D Data Generation for Robotic Perception

A procedural synthetic RGB-D data-generation and geometric-perception pipeline for a manufacturing scene containing aluminium-profile proxies and mounting brackets. The project investigates how selected depth-measurement imperfections affect downstream pose estimation.

Developed as part of a doctoral researcher application assignment in humanoid robotics, 3D perception and embodied AI.

## Project at a glance

The pipeline comprises:

1. Procedural scene and randomized-pose dataset generation with BlenderProc.
2. Semantic and geometric ground-truth export.
3. A phenomenological, D455-inspired depth-corruption model.
4. Point-cloud reconstruction and geometric pose estimation.
5. Randomized-pose experiments, corruption ablations and sensitivity analyses.

The reference RGB-D camera is the **Intel RealSense D455**. It was selected as a practical, well-documented stereo reference platform. The depth model is phenomenological: its parameters have **not** been calibrated against a physical D455, and no physical-camera validation is claimed.

## Important scope and limitations

- The long aluminium-profile target is represented by a **rectangular cuboid proxy**, not a detailed T-slot extrusion model. Some scene details are separate geometry and are not part of the target point cloud.
- The randomized dataset varies target yaw and horizontal position while keeping the camera fixed and target height constant.
- The downstream estimators use ground-truth semantic segmentation and camera parameters. The CAD-informed estimator additionally uses assumed target dimensions.
- The experiments evaluate geometric pose-estimation behaviour; they do not constitute autonomous robot grasping or physical manipulation trials.
- The depth corruption model approximates selected effects (disparity perturbation and quantisation, range-dependent missing depth, boundary failures, and optional spatially correlated disparity noise). It does not reproduce the complete RealSense stereo-matching and post-processing pipeline.

These qualifications are intentional and should be considered when interpreting the results.

## Software environment

The development environment used:

- Python 3.10
- Blender 4.2.1 LTS
- BlenderProc 2.8.0
- Open3D 0.20.0
- NumPy 2.2.6

Blender is a separate application; it is not installed by `requirements.txt`. The Python dependencies used directly by the project are listed there.

## Installation

Create and activate a Python 3.10 virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install Blender 4.2.1 LTS separately. BlenderProc 2.8.0 provides the `blenderproc` command used to run the rendering scripts. Depending on your system, its first run may need to download or configure a compatible Blender runtime.

Run all commands below from the repository root.

## Repository structure

```text
src/
  sensor/          # Depth-corruption model
  experiments/     # Geometric pose-estimation methods and metrics
scripts/            # Scene generation, experiments, audits and plotting
examples/           # Small representative dataset subset
results/            # Selected CSV/JSON experiment outputs
report/
  figures/          # Figures used for analysis and reporting
docs/               # Supporting project documentation
```

The full generated datasets and intermediate rendering outputs are not committed. They are written under `outputs/`, which is excluded by `.gitignore`. Three representative HDF5 frames and their pose metadata are provided under `examples/`.

## Reproducing the main experiments

### 1. Generate the randomized RGB-D dataset

```bash
blenderproc run scripts/create_randomized_pose_dataset.py
```

This generates 200 randomized samples under:

```text
outputs/randomized_pose_dataset/
```

Each rendered frame is stored as an individual HDF5 file, with pose and camera metadata in `pose_labels.json`.

### 2. Run the randomized OBB pose experiment

```bash
python -m scripts.run_randomized_pose_experiment
```

This evaluates the PCA-based oriented-bounding-box (OBB) estimator on ideal and fully corrupted depth. Results are written under `outputs/randomized_pose_experiment/`.

### 3. Run the OBB corruption ablation

```bash
python -m scripts.run_randomized_pose_ablation
```

This compares ideal depth, disparity-only corruption, dropout-only corruption, and full corruption. Results are written under `outputs/randomized_pose_ablation/`.

### 4. Run the spatial-correlation sensitivity experiment

```bash
python -m scripts.run_correlation_scale_sweep
```

This compares independent disparity noise with spatially correlated noise at several correlation scales. Results are written under `results/correlation_scale_sweep/`.

### 5. Run the CAD-informed end-face experiment

```bash
python -m scripts.run_cad_end_experiment
```

This evaluates the CAD-informed end-face estimator under ideal and corrupted depth. Results are written under `results/cad_end_estimator/`.

The experiments above expect the randomized dataset from step 1 to exist. They can take a substantial amount of time, depending on CPU, GPU, and rendering configuration.

## Additional scripts

The `scripts/` directory also contains diagnostic and analysis tools for:

- depth-corruption behaviour and parameter sensitivity;
- stereo-view geometric visibility and occlusion audits;
- normal-estimation and end-face visibility diagnostics;
- OBB and CAD-estimator failure analysis;
- paired statistical comparisons and figure generation.

Most scripts are intended to be run from the repository root using module syntax, for example:

```bash
python -m scripts.analyse_obb_cross_axis
```

Check the relevant script before running it: some diagnostic scripts expect intermediate files generated by an earlier experiment.

## Included data and results

- `examples/randomized_pose_dataset/`: three representative HDF5 frames and the associated pose metadata.
- `results/`: selected experimental summaries, per-trial CSV files, per-pose aggregates and diagnostic outputs.
- `report/figures/`: generated figures.

The examples are a small inspection subset, not the complete 200-frame dataset. Regenerate the full dataset using the command above.

## Report

The final application report is distributed as a PDF at `report/report.pdf`. The LaTeX source and bibliography are not included in this repository.

## Copyright and usage

Copyright (c) 2026 João Zenário. All rights reserved.

This repository is publicly available for academic review and evaluation purposes. No licence is granted for the reproduction, modification, redistribution, or commercial use of this source code or its associated materials without prior written permission from the copyright holder.

Public availability on GitHub does not constitute a general grant of rights to reuse this work. GitHub's Terms of Service apply to viewing and forking the public repository.

If you wish to use any part of this work beyond academic review and evaluation, please contact the copyright holder for permission.
