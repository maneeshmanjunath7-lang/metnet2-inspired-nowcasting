# train_radolan.py
import torch
from torch.utils.data import DataLoader
from mini_metnet_demo import MiniMetNet
from radolan_dataset import RadolanSequenceDataset

# 1) Point to your data sources (folders and/or .tar files)
SOURCES = [
    r"C:\Maneesh\Mission Germany\TUM Aerospace\Studies\Semester Thesis\Weather Forecasting\Data\RADOLAN\RW202412",
    # r"C:\path\to\another_day.tar",
]

# 2) Build dataset
dataset = RadolanSequenceDataset(
    sources=SOURCES,
    time_steps=6,             # past frames T
    target_offset=1,          # predict next frame
    crop_size=(64, 64),       # match your model
    stride=1,
    cache_dir=r"C:\temp\radolan_cache",  # optional, speeds up later runs
    normalization="log1p",
    product_hint="RW",        # filter members containing 'RW' (optional)
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# 3) Model / Loss / Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = MiniMetNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4) Train a few steps to validate end-to-end
model.train()
for epoch in range(1, 3):
    running = 0.0
    for x, y in loader:
        # x: (B, T, 1, H, W) -> model expects (B,T,C,H,W)
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    avg = running / len(loader)
    print(f"Epoch {epoch} | Avg Loss: {avg:.4f}")
    torch.save(model.state_dict(), f"radolan_epoch_{epoch}.pt")
