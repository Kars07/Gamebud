# 🎮 Gamebud-Nano: Efficient Visual Control from NitroGen Data

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![NVIDIA](https://img.shields.io/badge/Dataset-NitroGen-green) ![Status](https://img.shields.io/badge/Hardware-RTX_4050_(6GB)-blue)

**A lightweight Vision-Language-Action (VLA) agent trained on a curated subset of NVIDIA's NitroGen dataset, optimized for consumer hardware.**

##Overview
[cite_start]Large-scale foundation models like **NitroGen** are trained on 40,000 hours of video[cite: 10, 24], requiring massive compute clusters. **Gamebud-Nano** demonstrates that effective Visual Behavior Cloning (BC) can be achieved on a **single consumer GPU (6GB VRAM)** by leveraging smart data curation and efficient architectures.

This project implements a complete data-to-policy pipeline:
1.  [cite_start]**"Surgical" Data Extraction:** Streaming specific Parquet shards from Hugging Face without downloading the full 1.7TB dataset[cite: 26, 32].
2.  **Visual-Motor Synchronization:** Aligning raw controller telemetry with 60 FPS gameplay video.
3.  **Nano-VLA Policy:** A ResNet-18 backbone with a multi-head policy (Continuous Joysticks + Binary Buttons) that mimics human reaction times.

## 🎥 Results
**Agent Reaction Test (Behavior Cloning)**
The agent (Red Dots) predicts joystick movements based *only* on the visual feed.
> *Observation: The agent demonstrates "instant" zero-lag mimicry of the human player's reflexes, successfully learning the latent dynamics of the game.*

![Agent Demo](agent_replay.mp4)
*(Run `visualize_agent.py` to generate this visualization)*

## Technical Architecture

### 1. Data Engineering ("The Sniper Method")
[cite_start]Instead of downloading the full NitroGen dataset, this pipeline uses HTTP range requests to extract specific `actions_raw.parquet` files and their corresponding `metadata.json`[cite: 45, 53].
* **Input:** NitroGen Shard ID.
* **Process:** Stream Parquet -> Extract Telemetry -> Download Video Crop (`yt-dlp`).
* **Output:** Synchronized Tensor Dataset (Images + Action Labels).

### 2. Model: Gamebud-Nano
A unified neural policy designed to fit in <4GB VRAM.
* **Visual Encoder:** ResNet-18 (ImageNet Weights removed for "Tabula Rasa" learning).
* **Policy Head:**
    * **Latent:** 512 $\to$ 256 (ReLU + Dropout).
    * **Head A (Joysticks):** `Tanh` activation (Regresses $x, y$ to $[-1, 1]$).
    * **Head B (Buttons):** `BCEWithLogits` (Classifies 14 button states).

## Installation

```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/Gamebud-Nano.git](https://github.com/YOUR_USERNAME/Gamebud-Nano.git)
cd Gamebud-Nano

# Install dependencies
pip install torch torchvision opencv-python pandas pyarrow yt-dlp huggingface_hub
