import time

import pandas as pd
import pygame

# --- CONFIG ---
OUTPUT_LOG = "my_data_actions.parquet"
FPS = 60

print("Initializing PS4 (Custom Map) Input Logger...")

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("❌ No joystick detected.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected: {joystick.get_name()}")

data_records = []
print("\n LOGGING STARTED! (Press 'Ctrl+C' to stop)")
print("   -> Mash 'A' (Button 0) 3 times to sync.")
print("   -> Record your gameplay (Press ALL buttons!).")

try:
    start_time = time.time()
    while True:
        loop_start = time.time()
        pygame.event.pump()

        # --- CUSTOM MAPPING FOR YOUR CONTROLLER ---

        # 1. Triggers (Axes 4/5)
        l2_val = joystick.get_axis(4) if joystick.get_numaxes() > 4 else -1.0
        r2_val = joystick.get_axis(5) if joystick.get_numaxes() > 5 else -1.0
        l2_pressed = 1 if l2_val > -0.5 else 0
        r2_pressed = 1 if r2_val > -0.5 else 0

        # 2. Bumpers (You found these are 9 and 10)
        l1_pressed = joystick.get_button(9) if joystick.get_numbuttons() > 9 else 0
        r1_pressed = joystick.get_button(10) if joystick.get_numbuttons() > 10 else 0

        # 3. D-Pad (You found these are 11, 12, 13, 14)
        # Check specific mappings from your logs:
        # Up=11, Down=12, Left=13, Right=14 (Adjust if order differs)
        dpad_up = joystick.get_button(11) if joystick.get_numbuttons() > 11 else 0
        dpad_down = joystick.get_button(12) if joystick.get_numbuttons() > 12 else 0
        dpad_left = joystick.get_button(13) if joystick.get_numbuttons() > 13 else 0
        dpad_right = joystick.get_button(14) if joystick.get_numbuttons() > 14 else 0

        record = {
            # Joysticks
            "j_left_x": joystick.get_axis(0),
            "j_left_y": joystick.get_axis(1),
            "j_right_x": joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0,
            "j_right_y": joystick.get_axis(3) if joystick.get_numaxes() > 3 else 0.0,
            # Face Buttons (Standard 0-3)
            "south": joystick.get_button(0),
            "east": joystick.get_button(1),
            "west": joystick.get_button(2),
            "north": joystick.get_button(3),
            # Bumpers (Custom Mapped)
            "left_shoulder": l1_pressed,
            "right_shoulder": r1_pressed,
            # Triggers (Axis Mapped)
            "left_trigger": l2_pressed,
            "right_trigger": r2_pressed,
            "start": joystick.get_button(9)
            if joystick.get_numbuttons() > 9
            else 0,  # Usually 9/option
            # D-Pad (Button Mapped)
            "dpad_left": dpad_left,
            "dpad_right": dpad_right,
            "dpad_up": dpad_up,
            "dpad_down": dpad_down,
        }

        data_records.append(record)

        # FPS Limiter
        diff = time.time() - loop_start
        wait = max(0, (1 / FPS) - diff)
        time.sleep(wait)

except KeyboardInterrupt:
    print("\nLogging Stopped.")

# --- SAVE ---
if not data_records:
    print("No data recorded.")
    exit()

df = pd.DataFrame(data_records)

# Columns Check
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

df = df[required_cols]
df.to_parquet(OUTPUT_LOG)
print(f"Saved Fixed Log v3: {OUTPUT_LOG}")
