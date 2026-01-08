import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision.models as models
import numpy as np
import time


# CONFIG
BATCH_SIZE = 32        # Small batch for 6GB VRAM
LEARNING_RATE = 1e-4   # Standard BC rate
EPOCHS = 50            # Quick training for demonstration
DEVICE = "cuda"

