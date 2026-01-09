import pandas as pd
import numpy as np
import cv2
import torch
import os

# CONFIG
PARQUET_FILE = "training_actions.parquet"
VIDEO_FILE = "training_clip.mp4"
OUTPUT_FILE = "processed_dataset.pt"

print(f"Processing {PARQUET_FILE}...")

# 1. Load Actions & Flatten Joysticks
df = pd.read_parquet(PARQUET_FILE)

# The "List Exploder" Hack
# NitroGen stores joysticks as lists [x, y]. We need flat columns.
print("   -> Flattening joystick vectors...")

# Extract lists into separate columns
# j_left -> j_left_x, j_left_y
left_joy = pd.DataFrame(df['j_left'].tolist(), columns=['j_left_x', 'j_left_y'])
right_joy = pd.DataFrame(df['j_right'].tolist(), columns=['j_right_x', 'j_right_y'])

# Drop original list columns and attach new flat ones
df = df.drop(columns=['j_left', 'j_right'])
df = pd.concat([df, left_joy, right_joy], axis=1)

# Convert all buttons to float (0.0 or 1.0) for the neural net
# We drop 'back', 'guide', 'start' as they are menu buttons, not gameplay.
cols_to_drop = ['back', 'guide', 'start']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Normalize remaining data to float32
action_tensor = torch.tensor(df.values, dtype=torch.float32)
print(f"Actions Shape: {action_tensor.shape} (Frames, Controls)")


# 2. Process Video (Resize to 64x64 for 4050 Speed)
print(f"Processing {VIDEO_FILE}...")
cap = cv2.VideoCapture(VIDEO_FILE)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize to 64x64 (standard for efficient RL/BC)
    frame = cv2.resize(frame, (64, 64))
    
    # Convert BGR (OpenCV) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values (0-255 -> 0.0-1.0)
    frame = frame.astype(np.float32) / 255.0
    
    # Move Channel to first dimension: (H, W, C) -> (C, H, W)
    # PyTorch expects Channels First
    frame = np.transpose(frame, (2, 0, 1))
    
    frames.append(frame)

cap.release()
video_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
print(f"Video Shape: {video_tensor.shape} (Frames, C, H, W)")

# 3. Synchronization Check
# Video FPS might slightly differ from Action FPS. We truncate to the shorter one.
min_len = min(len(action_tensor), len(video_tensor))
action_tensor = action_tensor[:min_len]
video_tensor = video_tensor[:min_len]

print(f"Synced Length: {min_len} frames")

# 4. Save Final Dataset
torch.save({
    "states": video_tensor,
    "actions": action_tensor,
    "columns": df.columns.tolist() # Save column names so we know which button is which
}, OUTPUT_FILE)

print(f"\n Saved to {OUTPUT_FILE}")
print("Ready for Training!")