import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch._inductor.config import batch_fusion
from torch.utils.data import DataLoader, TensorDataset, random_split

# CONFIG
BATCH_SIZE = 32  # Small batch for 6GB VRAM
LEARNING_RATE = 1e-4  # Standard BC rate
EPOCHS = 50  # Quick training for demonstration
DEVICE = "cuda"

print(f"Initializing Gamebud Training on {DEVICE}...")


# Model Architecture
class GamebudNano(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # ResNet18 is the visual brain
        # We modify the first layer to accept 64x64 cleanly if needed
        # but standard ResNet adapts well enough
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace the final fully connected layer
        # ResNet output is 512 features
        self.backbone.fc = nn.Identity()  # Remove the ImageNet classifier

        # Define our Action Heads
        # We have 18 outputs in total
        # But we want to treat Joysticks (4 float) and Buttons (14 Binary) differently?

        # Shared Latent Layer
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1)
        )

        # Head A: Joysticks (Continuous -1 to 1) -> values
        self.joystick_head = nn.Linear(256, 4)

        # Head B: Buttons (Binary 0 or 1) -> 14 values
        self.button_head = nn.Linear(256, 14)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.policy_head(features)

        joysticks = torch.tanh(self.joystick_head(latent))  # Force between -1 and 1
        buttons = self.button_head(latent)  # Logits

        return joysticks, buttons


# Load the Data
print("Loading Data...")
data = torch.load("processed_dataset.pt")
inputs = data["states"]  # (N, 3, 64, 64)
targets = data["actions"]  # (N, 18)

# Separate targets into Joysticks (last 4 cols) and Buttons (first 14 cols)
# Based on previous output: Joysticks were appended at the END.
# So columns 0-13 are Buttons, 14-17 are Joysticks.
target_buttons = targets[:, :14]
target_joysticks = targets[:, 14:]

dataset = TensorDataset(inputs, target_joysticks, target_buttons)

# Split 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

print(f" Data Loaded: {len(train_data)} Train, {len(val_data)} Val")

# Training Setup
model = GamebudNano(output_dim=18).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss Functions
# MSE for joysticks (Precision needed)
criterion_joy = nn.MSELoss()
# BCE for Buttons (Did i press it? Yes/No)
criterion_btn = nn.BCEWithLogitsLoss()

# Training Loop
print("\n Starting Training...")
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

        # Forward pass
        pred_joy, pred_btn = model(imgs)

        # Calculate Loss(Weighted Sum)
        loss_joy = criterion_joy(pred_joy, true_joy)
        loss_btn = criterion_btn(pred_btn, true_btn)

        total_loss = loss_joy + loss_btn

        # Backward pass
        total_loss.backward()

        # Update weights
        optimizer.step()

        train_loss += total_loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, true_joy, true_btn in val_loader:
            imgs, true_joy, true_btn = (
                imgs.to(DEVICE),
                true_joy.to(DEVICE),
                true_btn.to(DEVICE),
            )
            pred_joy, pred_btn = model(imgs)
            val_loss += criterion_joy(pred_joy, true_joy) + criterion_btn(
                pred_btn, true_btn
            )

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
        )

# Saving the model
print("\n Saving the model...")
torch.save(model.state_dict(), "gamebud_model.pth")
print("Model saved successfully!")
print(f"Total Time: {time.time() - start_time:.1f}s")
