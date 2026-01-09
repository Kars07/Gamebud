import time

import pandas as pd
import pygame

# --- CONFIG ---
OUTPUT_LOG = "my_data_actions.parquet"
FPS = 30  # We will try to match standard video FPS

print("Initializing Windows Input Logger...")

pygame.init()
pygame.joystick.init()

# Check for controller
if pygame.joystick.get_count() == 0:
    print("❌ No joystick detected on Windows.")
    input("Press Enter to exit...")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller Connected: {joystick.get_name()}")
print("   -> READY.")

data_records = []
print("\n LOGGING STARTED! (Press 'Ctrl+C' in this window to stop)")

try:
    start_time = time.time()
    while True:
        loop_start = time.time()
        pygame.event.pump()  # Read hardware inputs

        # Capture Inputs
        record = {
            # Joysticks
            "j_left_x": joystick.get_axis(0),
            "j_left_y": joystick.get_axis(1),
            "j_right_x": joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0,
            "j_right_y": joystick.get_axis(3) if joystick.get_numaxes() > 3 else 0.0,
            # Buttons (Generic Mapping)
            "south": joystick.get_button(0),
            "east": joystick.get_button(1),
            "west": joystick.get_button(2),
            "north": joystick.get_button(3),
            "left_shoulder": joystick.get_button(4),
            "right_shoulder": joystick.get_button(5),
            # Try to grab triggers as buttons (often 6/7 on Windows too)
            "left_trigger": joystick.get_button(6)
            if joystick.get_numbuttons() > 6
            else 0,
            "right_trigger": joystick.get_button(7)
            if joystick.get_numbuttons() > 7
            else 0,
            "start": joystick.get_button(9) if joystick.get_numbuttons() > 9 else 0,
            # D-Pad
            "dpad_left": 0,
            "dpad_right": 0,
            "dpad_up": 0,
            "dpad_down": 0,
        }

        # D-Pad Logic
        if joystick.get_numhats() > 0:
            hat = joystick.get_hat(0)
            if hat[0] == -1:
                record["dpad_left"] = 1
            if hat[0] == 1:
                record["dpad_right"] = 1
            if hat[1] == 1:
                record["dpad_up"] = 1
            if hat[1] == -1:
                record["dpad_down"] = 1

        data_records.append(record)

        # 30 FPS Limiter
        diff = time.time() - loop_start
        wait = max(0, (1 / FPS) - diff)
        time.sleep(wait)

except KeyboardInterrupt:
    print("\n Logging Stopped.")

# --- SAVE ---
if not data_records:
    print(" No data recorded.")
    exit()

df = pd.DataFrame(data_records)

# Ensure columns match training script
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
        df[col] = 0.0

# Reorder
df = df[required_cols]
df.to_parquet(OUTPUT_LOG)
print(f"Saved: {OUTPUT_LOG}")
