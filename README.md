# NuScenes Trajectory Prediction: Constant Velocity Baseline

## Project Overview
This repository contains a robust, physics-based baseline for **Problem Statement 1: Intent & Trajectory Prediction**. 

In autonomous vehicle perception, deep learning architectures (LSTMs, Transformers, BEV-Formers) are the industry standard. However, they frequently fail to outperform simple kinematic models over short time horizons (e.g., 3 seconds) when predicting pedestrian and cyclist behavior. Before engineering a highly complex, computationally heavy neural network, it is critical to establish a rigorous mathematical lower bound. 

This project establishes that baseline. It implements a deterministic kinematic model to predict the future coordinates of non-motorized agents, optimizing for Average Displacement Error (ADE) and Final Displacement Error (FDE) without the overhead of neural network inference.

## Model Architecture
This solution bypasses parametric neural networks in favor of a **Constant Velocity (CV) Kinematic Model** augmented with rotational matrices for multi-modal generation. 

The architecture is purely mathematical and executes in the following pipeline:
1. **State Extraction:** Parses a 2-second coordinate history (at 2Hz) for agents classified as `human` or `cycle`.
2. **Velocity Vectorization:** Calculates the agent's instantaneous velocity vector $v$ using the displacement between the current position $p_t$ and the previous timestep $p_{t-1}$.
3. **Multi-Modal Projection:** To satisfy the multi-modal requirement (predicting the 3 most likely paths), the model generates a primary kinematic projection and applies a 2D rotation matrix to the velocity vector to simulate realistic lateral drift over the next 3 seconds (6 timesteps).
   - **Path 1 (Probability 0.60):** True Constant Velocity (straight-line projection).
   - **Path 2 (Probability 0.20):** $+5^\circ$ lateral drift constraint.
   - **Path 3 (Probability 0.20):** $-5^\circ$ lateral drift constraint.

## Dataset Used
- **nuScenes (`v1.0-mini`)**
- The model specifically filters the `sample_annotation` tables to isolate the bounding box translation coordinates of pedestrians and cyclists, ignoring raw LiDAR/Camera data in favor of pure spatial sequences.

## Setup & Installation Instructions
The environment requires Python 3.8+ and minimal dependencies to ensure rapid, edge-compatible execution.

1. Clone this repository.
2. Ensure the nuScenes `v1.0-mini` dataset is extracted locally (default path: `./data`).
3. Install the required libraries:
```bash
pip install nuscenes-devkit numpy
