# Gamebud-Nano: End-to-End Visual Behavior Cloning

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![License](https://img.shields.io/badge/License-MIT-blue) ![Status](https://img.shields.io/badge/Status-Production_Ready-green)

**A lightweight Vision-Language-Action (VLA) agent that plays video games by "watching" the screen. Trained on a single consumer GPU.**

## Overview
**Gamebud-Nano** is a Visual Behavior Cloning (BC) agent designed to replicate human gameplay strategies with sub-frame latency. Unlike massive foundation models requiring H100 clusters, this project demonstrates that **ResNet-18** architectures can achieve high-fidelity control when paired with:
1.  **High-Quality Data:** A custom 60FPS telemetry logger.
2.  **Weighted Loss Functions:** To handle severe class imbalance in sparse actions (e.g., Triggers).
3.  **Hardware Normalization:** Custom drivers to map diverse controller schemas (PS4/Xbox) to a unified action space.

![Demo](demo.gif)
*(The agent predicting actions (Green) based purely on visual input)*

## Technical Architecture

### 1. The Model (Gamebud-Nano)
A specialized Convolutional Neural Network (CNN) tailored for real-time control.
* **Backbone:** ResNet-18 (Weights initialized from scratch or ImageNet).
* **Policy Head A (Joysticks):** Regresses continuous values `[-1, 1]` using `MSELoss`.
* **Policy Head B (Buttons):** Classifies 14 binary inputs using **Weighted BCEWithLogitsLoss**.
* **Optimization:** Weights are dynamically scaled (up to 50x) based on action frequency in the training set to solve the "Lazy Agent" problem.

### 2. The Data Pipeline ("Golden Dataset")
Standard datasets often lack specific hardware mappings. This repo includes a full data engineering suite:
* **`windows_logger_60fps.py`:** A custom Python driver that bypasses standard OS constraints to read raw Axis/Button data from PS4/Xbox controllers at 60Hz.
* **`prepare_custom.py`:** A synchronization tool that aligns variable-rate logs with variable-rate gameplay video, correcting for drift and FPS mismatches.

## Installation

```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/Gamebud-Nano.git](https://github.com/YOUR_USERNAME/Gamebud-Nano.git)
cd Gamebud-Nano

# Install dependencies (WSL/Linux)
pip install torch torchvision opencv-python pandas pyarrow pygame

# Note: For recording data, you must install 'pygame' on Windows as well.
