# Synthetic RGB-D Data Generation for Robotic Perception

A procedural synthetic RGB-D data-generation and perception-evaluation
pipeline for a manufacturing scene containing aluminium extrusion
profiles and mounting brackets.

Developed as part of a doctoral researcher application assignment in
humanoid robotics, 3D perception and embodied AI.

## Reference camera

Intel RealSense D455.

The camera was selected using a qualitative weighted multi-criteria
assessment. The depth corruption model uses D455-informed assumptions;
it has not been calibrated against a physical D455.

## Pipeline

BlenderProc scene generation
-> labelled RGB-D rendering
-> phenomenological depth corruption
-> point-cloud reconstruction
-> PCA-based oriented bounding-box pose estimation
-> quantitative evaluation and corruption ablation.

## Software environment

- Python 3.10
- Blender 4.2.1 LTS
- BlenderProc 2.8.0
- Open3D 0.20.0

## Repository structure

- `src/`: sensor and perception modules
- `scripts/`: scene generation, dataset generation, experiments and audits
- `configs/`: project configuration
- `report/`: LaTeX report, bibliography and figures
- `results/`: selected experimental CSV and JSON outputs
- `examples/`: representative synthetic dataset samples

## Generate the manufacturing scene

```bash
blenderproc run scripts/create_manufacturing_scene.py
```
## Copyright and usage

Copyright (c) 2026 João Zenário. All rights reserved.

This repository is publicly available for academic review and evaluation purposes. No licence is granted for the reproduction, modification, redistribution, or commercial use of this source code or its associated materials without prior written permission from the copyright holder.

Public availability on GitHub does not constitute a general grant of rights to reuse this work. GitHub's Terms of Service apply to viewing and forking the public repository.

If you wish to use any part of this work beyond academic review and evaluation, please contact the copyright holder for permission.
