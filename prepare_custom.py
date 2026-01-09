import os

import cv2
import numpy as np
import pandas as pd
import torch

# CONFIG
PARQUET_FILE = "training_actions.parquet"
VIDEO_FILE = "training_clip.mp4"
OUTPUT_FILE = "processed_dataset.pt"

print(f" Processing Custom Golden Dataset...")

# --- 1. Load Actions ---
df = pd.read_parquet(PARQUET_FILE)

# Drop menu buttons that we don't train on
cols_to_drop = ["back", "guide", "start"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Force correct column order for the model
# (14 Buttons first, then 4 Joysticks)
button_cols = [
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "east",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "north",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "west",
]
joy_cols = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"]

# Reorder and verify
try:
    df = df[button_cols + joy_cols]
except KeyError as e:
    print(f"❌ Error: Missing columns in parquet file! {e}")
    exit()

action_tensor = torch.tensor(df.values, dtype=torch.float32)
print(f"Actions Loaded: {len(df)} frames")

# --- 2. Process Video ---
print(f"   Reading video {VIDEO_FILE}...")
cap = cv2.VideoCapture(VIDEO_FILE)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 64x64 for the AI
    frame = cv2.resize(frame, (64, 64))
    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # HWC -> CHW (PyTorch format)
    frame = np.transpose(frame, (2, 0, 1))
    frames.append(frame)

cap.release()

# Normalize pixel values (0-255 -> 0.0-1.0)
video_tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0
print(f"Video Loaded: {len(frames)} frames")

# --- 3. Smart Synchronization ---
n_actions = len(action_tensor)
n_video = len(video_tensor)

# Heuristic: If video is ~2x longer than actions, it's likely 60fps vs 30fps
# if n_video > 1.8 * n_actions:
#     print("Detected FPS Mismatch (Video ~60fps, Logs 30fps). Downsampling video...")
#     video_tensor = video_tensor[::2]  # Take every 2nd frame
#     print(f"   -> New Video Length: {len(video_tensor)}")

# Since both are 60 FPS now, just sync lengths directly
print(f"   Actions: {n_actions}, Video: {n_video}")
min_len = min(n_actions, n_video)
action_tensor = action_tensor[:min_len]
video_tensor = video_tensor[:min_len]

# Truncate to the shorter length to align them
# min_len = min(len(action_tensor), len(video_tensor))
# action_tensor = action_tensor[:min_len]
# video_tensor = video_tensor[:min_len]

print(f"Final Synced Dataset: {min_len} frames")

# --- 4. Save ---
torch.save(
    {"states": video_tensor, "actions": action_tensor, "columns": df.columns.tolist()},
    OUTPUT_FILE,
)
print(f"Saved to {OUTPUT_FILE}")
