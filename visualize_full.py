import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --- CONFIG ---
MODEL_PATH = "gamebud_weighted.pth"
VIDEO_PATH = "training_clip.mp4"
OUTPUT_PATH = "agent_full_replay.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The column order from your prepare_dataset.py script
BUTTON_MAP = {
    "dpad_down": 0,
    "dpad_left": 1,
    "dpad_right": 2,
    "dpad_up": 3,
    "east": 4,
    "left_shoulder": 5,
    "left_thumb": 6,
    "left_trigger": 7,
    "north": 8,
    "right_shoulder": 9,
    "right_thumb": 10,
    "right_trigger": 11,
    "south": 12,
    "west": 13,
}

print(f"Initializing Full Visualization on {DEVICE}...")


# --- 1. Model Definition ---
class GamebudNano(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
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
        return torch.tanh(self.joystick_head(latent)), self.button_head(latent)


# Load Model
model = GamebudNano(18).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded.")
except FileNotFoundError:
    print("Model not found!")
    exit()
model.eval()


# --- 2. Drawing Helpers ---
def draw_button_circle(img, cx, cy, label, active, size=10):
    # Color: Green if active, Grey if inactive
    color = (0, 255, 0) if active else (50, 50, 50)
    thickness = -1 if active else 2
    cv2.circle(img, (cx, cy), size, color, thickness)
    cv2.circle(img, (cx, cy), size, (200, 200, 200), 1)
    if label:
        cv2.putText(
            img,
            label,
            (cx - 5, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )


def draw_rect_btn(img, x, y, w, h, label, active):
    color = (0, 255, 0) if active else (50, 50, 50)
    thickness = -1 if active else 2
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    if label:
        cv2.putText(
            img,
            label,
            (x + 2, y + h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )


# --- 3. Video Loop ---
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("Rendering HUD (with Debug Prints)...")

frame_count = 0  # <--- Initialize counter here

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    small = cv2.resize(frame, (64, 64))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        / 255.0
    )

    with torch.no_grad():
        joy_logits, btn_logits = model(tensor)
        # Apply Sigmoid to buttons to get 0.0-1.0 probability
        btn_probs = torch.sigmoid(btn_logits).cpu().numpy()[0]
        joy = joy_logits.cpu().numpy()[0]

    # --- DEBUGGING ---
    r2_prob = btn_probs[BUTTON_MAP["right_trigger"]]
    l2_prob = btn_probs[BUTTON_MAP["left_trigger"]]

    # Print to terminal if the model is even *thinking* about pressing triggers (> 10%)
    if r2_prob > 0.1 or l2_prob > 0.1:
        print(f"Frame {frame_count} | L2: {l2_prob:.3f} | R2: {r2_prob:.3f}")

    # --- DRAW HUD ---
    # Draw Background Bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 120), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 1. Joysticks
    lx, ly = int(90 + joy[0] * 30), int(height - 60 - joy[1] * 30)
    rx, ry = int(250 + joy[2] * 30), int(height - 60 - joy[3] * 30)

    cv2.circle(frame, (90, height - 60), 30, (50, 50, 50), 2)
    cv2.circle(frame, (250, height - 60), 30, (50, 50, 50), 2)
    cv2.circle(frame, (lx, ly), 10, (0, 0, 255), -1)
    cv2.circle(frame, (rx, ry), 10, (0, 0, 255), -1)

    # 2. D-Pad & Buttons (Threshold 0.25 for higher sensitivity)
    THRESHOLD = 0.25

    base_x, base_y = 40, height - 60
    draw_button_circle(
        frame, base_x, base_y - 15, "U", btn_probs[BUTTON_MAP["dpad_up"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x, base_y + 15, "D", btn_probs[BUTTON_MAP["dpad_down"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x - 15, base_y, "L", btn_probs[BUTTON_MAP["dpad_left"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x + 15, base_y, "R", btn_probs[BUTTON_MAP["dpad_right"]] > THRESHOLD
    )

    base_x, base_y = 350, height - 60
    draw_button_circle(
        frame, base_x, base_y - 15, "Y", btn_probs[BUTTON_MAP["north"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x, base_y + 15, "A", btn_probs[BUTTON_MAP["south"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x - 15, base_y, "X", btn_probs[BUTTON_MAP["west"]] > THRESHOLD
    )
    draw_button_circle(
        frame, base_x + 15, base_y, "B", btn_probs[BUTTON_MAP["east"]] > THRESHOLD
    )

    # 3. Triggers (Lower threshold to 0.20 to catch weak signals)
    draw_rect_btn(
        frame,
        20,
        height - 110,
        30,
        10,
        "L1",
        btn_probs[BUTTON_MAP["left_shoulder"]] > 0.20,
    )
    draw_rect_btn(
        frame,
        20,
        height - 125,
        30,
        10,
        "L2",
        btn_probs[BUTTON_MAP["left_trigger"]] > 0.20,
    )

    draw_rect_btn(
        frame,
        330,
        height - 110,
        30,
        10,
        "R1",
        btn_probs[BUTTON_MAP["right_shoulder"]] > 0.20,
    )
    draw_rect_btn(
        frame,
        330,
        height - 125,
        30,
        10,
        "R2",
        btn_probs[BUTTON_MAP["right_trigger"]] > 0.20,
    )

    out.write(frame)
    frame_count += 1  # <--- Increment counter

cap.release()
out.release()
print(f"Saved full visualization to {OUTPUT_PATH}")
