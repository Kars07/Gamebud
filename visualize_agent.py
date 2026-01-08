import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --- CONFIG ---
MODEL_PATH = "gamebud_model.pth"
VIDEO_PATH = "training_clip.mp4"
OUTPUT_PATH = "agent_replay.mp4"
DEVICE = "cuda"


class GamebudNano(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # We can use weights=None here because we load the state_dict immediately after
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.policy_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1)
        )
        self.joystick_head = nn.Linear(256, 4)
        self.button_head = nn.Linear(256, 14)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.policy_head(features)
        # Match training activation: Tanh for Joysticks, Raw Logits for Buttons
        return torch.tanh(self.joystick_head(latent)), self.button_head(latent)


# Load Model
model = GamebudNano(18).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model weights loaded.")
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found.")
    exit()

model.eval()

# Video Processing
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open {VIDEO_PATH}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("Rendering AI Overlay...")


def draw_joystick(img, x, y, cx, cy, label):
    # Draw Base (Grey Circle)
    cv2.circle(img, (cx, cy), 35, (50, 50, 50), -1)
    cv2.circle(img, (cx, cy), 35, (200, 200, 200), 2)

    # Draw Stick Position (Red Dot)
    # Model outputs -1.0 to 1.0. We scale this to +/- 30 pixels
    move_x = int(x * 30)
    move_y = int(y * 30)

    # Invert Y because screen coordinates go Down
    cv2.circle(img, (cx + move_x, cy + move_y), 12, (0, 0, 255), -1)

    # Label
    cv2.putText(
        img,
        label,
        (cx - 25, cy - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )


frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess for Model (Resize 64x64 -> Tensor)
    small_frame = cv2.resize(frame, (64, 64))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # (H, W, C) -> (C, H, W) -> Batch Dimension -> Normalize
    input_tensor = (
        torch.tensor(rgb_frame, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )

    # Inference
    with torch.no_grad():
        joy_pred, btn_pred = model(input_tensor)

    # Get Data back to CPU
    joy = joy_pred.cpu().numpy()[0]  # [Lx, Ly, Rx, Ry]

    #  DRAWING THE HUD
    # 1. Semi-transparent box at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, height - 150), (320, height - 20), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Draw Left Stick
    draw_joystick(frame, joy[0], joy[1], 90, height - 85, "L-Stick")

    # 3. Draw Right Stick
    draw_joystick(frame, joy[2], joy[3], 230, height - 85, "R-Stick")

    # 4. Draw Status Text
    cv2.putText(
        frame,
        "GAMEBUD AI ACTIVE",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "GAMEBUD AI ACTIVE",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    out.write(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"   -> Rendered {frame_count} frames...")

cap.release()
out.release()

print(f"\n SUCCESS! Video saved to: {OUTPUT_PATH}")
