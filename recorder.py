import os
import time

import cv2
import mss
import numpy as np
import pandas as pd
import pygame

# --- CONFIG ---
OUTPUT_VIDEO = "my_data_clip.mp4"
OUTPUT_LOG = "my_data_actions.parquet"
FPS = 30
RESIZE_TO = (64, 64)  # Match your training size immediately

print("Initializing Recorder...")

# 1. Init Controller
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("❌ No joystick detected! Plug in your controller.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller: {joystick.get_name()}")

# 2. Init Screen Capture
sct = mss.mss()
# Capture the primary monitor (Monitor 1)
monitor = sct.monitors[1]

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# We record at 64x64 to match the AI input (saves space too)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, RESIZE_TO)

# Storage
data_records = []
print("\n RECORDING STARTED! (Press 'Ctrl+C' in terminal to stop)")
print("   -> Go press every button on your controller now!")

start_time = time.time()
frame_count = 0

try:
    while True:
        loop_start = time.time()

        # A. Process Events (Must do this to read inputs)
        pygame.event.pump()

        # B. Capture Screen
        screenshot = np.array(sct.grab(monitor))
        # Remove Alpha channel (BGRA -> BGR)
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        # Resize for AI
        small_frame = cv2.resize(frame, RESIZE_TO)
        out.write(small_frame)

        # C. Capture Inputs (Normalize to match NitroGen format)
        # NitroGen: Joysticks are -1.0 to 1.0. Buttons are 0 or 1.

        # Standard Xbox/Generic Layout Map
        record = {
            "j_left_x": joystick.get_axis(0),
            "j_left_y": joystick.get_axis(1),
            "j_right_x": joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0,
            "j_right_y": joystick.get_axis(3) if joystick.get_numaxes() > 3 else 0.0,
            # Buttons (Map these indices to your controller if needed)
            "south": joystick.get_button(0),  # A
            "east": joystick.get_button(1),  # B
            "west": joystick.get_button(2),  # X
            "north": joystick.get_button(3),  # Y
            "left_shoulder": joystick.get_button(4),
            "right_shoulder": joystick.get_button(5),
            # Some controllers use Axes for triggers, some use Buttons.
            # We record buttons 6/7 for L2/R2 as a fallback.
            "left_trigger": joystick.get_button(6)
            if joystick.get_numbuttons() > 6
            else 0,
            "right_trigger": joystick.get_button(7)
            if joystick.get_numbuttons() > 7
            else 0,
            "start": joystick.get_button(9) if joystick.get_numbuttons() > 9 else 0,
            # D-Pad (Hat)
            "dpad_left": 0,
            "dpad_right": 0,
            "dpad_up": 0,
            "dpad_down": 0,
        }

        # Handle D-Pad (Hat)
        if joystick.get_numhats() > 0:
            hat = joystick.get_hat(0)  # (x, y)
            if hat[0] == -1:
                record["dpad_left"] = 1
            if hat[0] == 1:
                record["dpad_right"] = 1
            if hat[1] == 1:
                record["dpad_up"] = 1
            if hat[1] == -1:
                record["dpad_down"] = 1

        data_records.append(record)

        frame_count += 1

        # FPS Limiter
        diff = time.time() - loop_start
        wait = max(0, (1 / FPS) - diff)
        time.sleep(wait)

except KeyboardInterrupt:
    print("\n Stopping recording...")

# --- SAVE DATA ---
out.release()
df = pd.DataFrame(data_records)

# RENAME COLUMNS to match your training script exactly
# Your training script expects specific names, so we ensure they exist.
# We create dummy columns for any missing ones (like 'left_thumb')
required_cols = [
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
    "back",
    "guide",
    "start",
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0.0  # Fill missing with 0

# Reorder to match NitroGen structure
# Note: Your training script expects Joysticks at the END.
# Buttons first, then Joysticks.
cols_ordered = [
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
    "back",
    "guide",
    "start",
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
]
df = df[cols_ordered]

# Save
df.to_parquet(OUTPUT_LOG)

print(f"Saved Video: {OUTPUT_VIDEO} ({frame_count} frames)")
print(f"Saved Data: {OUTPUT_LOG}")
print("Ready to train on your own data!")
