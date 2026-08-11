import tarfile
import io
import torch
import torch.nn as nn
import torch.optim as optim
from model import MetNet2Mini
from data_loader import load_radolan_tensor

# Config
tar_path = "radolan_data/radolan_archive.tar.gz"  # path to your TAR file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open tar file and list its contents
with tarfile.open(tar_path, 'r') as tar:
    # List all .bin files sorted by name (assumed chronological)
    filenames = sorted([m.name for m in tar.getmembers() if m.name.endswith("dwd---bin")])
    print(f"Number of .bin files found: {len(filenames)}")
    print("First 5 filenames:", filenames[:5])

    # Prepare model and optimizer
    model = MetNet2Mini().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        # Iterate over sequences of length 7 (6 inputs + 1 target)
        for i in range(len(filenames) - 6):
            input_frames = []

            # Load 6 input frames
            for j in range(6):
                member = tar.getmember(filenames[i + j])
                f = tar.extractfile(member)
                data = load_radolan_tensor(io.BytesIO(f.read()))
                input_frames.append(data)

            x = torch.cat(input_frames, dim=1).to(device)  # shape [1, 6, H, W]

            # Load 7th frame as target
            member = tar.getmember(filenames[i + 6])
            f = tar.extractfile(member)
            raw_target = load_radolan_tensor(io.BytesIO(f.read())).to(device)  # [1,1,H,W]
            raw_target = raw_target.squeeze(0).squeeze(0)  # [H, W]

            target = torch.clamp((raw_target * 31).long(), 0, 31).unsqueeze(0)  # [1, H, W]

            # Forward + backward
            optimizer.zero_grad()
            out = model(x)  # [1, 32, H, W]
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
