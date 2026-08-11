# Updated train.py with real target from 7th radar frame
import torch
import torch.nn as nn
import torch.optim as optim
from model import MetNet2Mini
from data_loader import load_sequence, load_radolan_tensor

# Config
radolan_files = [f"radolan_data/frame{i}.bin" for i in range(6)]
target_file = "radolan_data/frame6.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load input (sequence of 6 hourly radar frames)
x = load_sequence(radolan_files).to(device)  # shape [1, 6, H, W]

# Load the 7th frame as target
raw_target = load_radolan_tensor(target_file).to(device)  # shape [1, 1, H, W]
raw_target = raw_target.squeeze(0).squeeze(0)  # shape [H, W]

# Convert rainfall intensities (0.0–1.0) to 32-class bins
target = torch.clamp((raw_target * 31).long(), 0, 31).unsqueeze(0)  # shape [1, H, W]

# Model setup
model = MetNet2Mini().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train loop
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    out = model(x)  # [1, 32, H, W]
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
