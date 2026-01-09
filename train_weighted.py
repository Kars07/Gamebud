import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- CONFIG ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cuda"

print(f"Initializing Weighted Training on {DEVICE}...")

#1. Load & Analyze Data
print("Loading Dataset...")
try:
    data = torch.load("processed_dataset.pt")
except FileNotFoundError:
    print("Error: 'processed_dataset.pt' not found.")
    sys.exit()
inputs = data["states"]  # (N, 3, 64, 64)
targets = data["actions"]  # (N, 18)

# Split Targets
target_buttons = targets[:, :14]  # First 14 are binary buttons
target_joysticks = targets[:, 14:]  # Last 4 are continuous joysticks

# CALCULATE WEIGHTS
# We count how many times each button was actually pressed
# If a button was pressed 10 times in 600 frames:
# Weight = (590 negatives) / (10 positives) = 59x penalty
num_samples = len(target_buttons)
num_pos = target_buttons.sum(dim=0)
num_neg = num_samples - num_pos

# Calculate weight: negative_count / positive_count
# We add 1e-5 to avoid dividing by zero if a button is NEVER pressed
pos_weights = num_neg / (num_pos + 1e-5)

print("\n Calculated Loss Weights (Sensitivity):")
cols = data["columns"][:14]  # Get names
for i, name in enumerate(cols):
    print(f"   - {name}: {pos_weights[i]:.1f}x")

# Move weights to GPU for the Loss Function
pos_weights = pos_weights.to(DEVICE)

# Prepare Dataloaders
dataset = TensorDataset(inputs, target_joysticks, target_buttons)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

if train_size == 0:
    train_size = len(dataset)
    val_size = 0  # Safety for tiny data
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE) if val_size > 0 else None


# 2. Model Architecture
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


model = GamebudNano(output_dim=18).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Loss Functions
criterion_joy = nn.MSELoss()
criterion_btn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# 4. Training Loop
print("\n Starting Weighted Training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, true_joy, true_btn in train_loader:
        imgs, true_joy, true_btn = (
            imgs.to(DEVICE),
            true_joy.to(DEVICE),
            true_btn.to(DEVICE),
        )

        optimizer.zero_grad()
        pred_joy, pred_btn = model(imgs)

        loss_joy = criterion_joy(pred_joy, true_joy)
        loss_btn = criterion_btn(pred_btn, true_btn)

        total_loss = loss_joy + loss_btn
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_loss / len(train_loader):.4f}"
        )
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

# 5. Save
torch.save(model.state_dict(), "gamebud_weighted.pth")
print("\nSaved model to 'gamebud_weighted.pth'")
