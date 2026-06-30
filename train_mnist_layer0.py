#!/usr/bin/env python3
"""
train_mnist_layer0.py -- trains a small MLP on MNIST and exports layer 0
(fc1: Linear(784->128)) weights and one test image for the pretrained-weights
milestone of Project Hydra.

Output files (same raw-float32 format as gen_layer0.py, real trained weights):
  data/input.bin          -- test image #0, flattened+normalized, 784 floats
  data/layer0_weights.bin -- fc1.weight [128x784 row-major] then fc1.bias [128]
                             total: (128*784 + 128) * 4 = 401,920 bytes
  data/expected_y.bin     -- expected fc1 output for input.bin, 128 floats
                             used by compare_output.py after the round trip

Run:  python train_mnist_layer0.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IN_DIM  = 784
OUT_DIM = 128
EPOCHS  = 2
LR      = 0.01
MOMENTUM = 0.9

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

os.makedirs("data", exist_ok=True)

print("Loading MNIST (downloads ~11 MB on first run to data/MNIST/raw/)...")
train_data = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64,   shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, OUT_DIM)   # layer we're exporting
        self.fc2 = nn.Linear(OUT_DIM, 10)

    def forward(self, x):
        x = x.view(-1, IN_DIM)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = MLP()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()

print(f"Training MLP ({IN_DIM}->{OUT_DIM}->10) for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch}/{EPOCHS}  avg loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        correct += (model(images).argmax(dim=1) == labels).sum().item()
print(f"Test accuracy: {correct}/10000 = {correct/100:.2f}%")

# --- export weights ---
W = model.fc1.weight.detach().cpu().float()  # [128, 784] OUT_DIM x IN_DIM row-major
b = model.fc1.bias.detach().cpu().float()    # [128]

with open("data/layer0_weights.bin", "wb") as f:
    f.write(W.numpy().astype("<f4").tobytes())
    f.write(b.numpy().astype("<f4").tobytes())
print(f"Wrote data/layer0_weights.bin  {(OUT_DIM*IN_DIM + OUT_DIM)*4} bytes"
      f"  (W[{OUT_DIM}x{IN_DIM}] row-major + b[{OUT_DIM}])")

# --- export test image #0 (already normalized by the DataLoader transform) ---
img_tensor, label = test_data[0]          # shape [1, 28, 28]
img_flat = img_tensor.view(IN_DIM).float()

with open("data/input.bin", "wb") as f:
    f.write(img_flat.numpy().astype("<f4").tobytes())
print(f"Wrote data/input.bin           {IN_DIM*4} bytes  (test image #0, label={label}, normalized)")

# --- compute expected fc1 output (raw linear, no ReLU) ---
# The worker computes y = W@x + b without activation; we match that here.
# Verification tolerance: max abs diff < 1e-4 expected due to different
# float32 accumulation order (BLAS vs. naive nested loop).
with torch.no_grad():
    y = model.fc1(img_flat.unsqueeze(0)).squeeze(0)  # [128]

with open("data/expected_y.bin", "wb") as f:
    f.write(y.numpy().astype("<f4").tobytes())
print(f"Wrote data/expected_y.bin      {OUT_DIM*4} bytes  (expected fc1 output, 128 floats)")

print(f"\nExpected y = fc1(x)  [{OUT_DIM} floats, no ReLU, test image #0 label={label}]:")
vals = [round(float(v), 6) for v in y]
# print 8 per line for readability
for i in range(0, OUT_DIM, 8):
    print(" ", vals[i:i+8])
print(f"\nmax |y|: {float(y.abs().max()):.6f}")
print("\nRun compare_output.py after the end-to-end test to verify max abs diff.")
