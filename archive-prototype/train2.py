import tarfile
import io
import torch
import torch.nn as nn
import torch.optim as optim
from model import MetNet2Mini
from data_loader import load_radolan_tensor
import time

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
tar_path = "radolan_data/radolan_archive.tar.gz"  # path to your TAR file

# Initialize model, optimizer, loss, scheduler
model = MetNet2Mini().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

epochs = 10  # Increase epochs to give scheduler room to work

with tarfile.open(tar_path, 'r') as tar:
    # Filter and sort files by name (adjust 'endswith' based on your file pattern)
    filenames = sorted([m.name for m in tar.getmembers() if m.name.endswith("dwd---bin")])
    print(f"Found {len(filenames)} files.")

    if len(filenames) < 7:
        raise RuntimeError("Not enough files to form sequences of 7 frames.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        start_time = time.time()

        for i in range(len(filenames) - 6):
            input_frames = []
            for j in range(6):
                f = tar.extractfile(filenames[i + j])
                data = load_radolan_tensor(io.BytesIO(f.read()))
                input_frames.append(data.to(device))  # Move input tensor to device

            x = torch.cat(input_frames, dim=1).to(device)  # shape [1, 6, H, W]

            f = tar.extractfile(filenames[i + 6])
            raw_target = load_radolan_tensor(io.BytesIO(f.read())).to(device)
            raw_target = raw_target.squeeze(0).squeeze(0)  # [H, W]
            target = torch.clamp((raw_target * 31).long(), 0, 31).unsqueeze(0).to(device)  # [1, H, W]

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else float('nan')
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # Step LR scheduler
        scheduler.step(avg_loss)

        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
