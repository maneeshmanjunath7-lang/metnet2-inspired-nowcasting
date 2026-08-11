# mini_metnet_demo.py
# A tiny, beginner-friendly MetNet-style model:
# Encoder (Conv2d) -> Temporal Aggregator (GRU) -> Decoder (Conv2d)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MiniMetNet(nn.Module):
    """
    Miniature MetNet-style model for precipitation nowcasting:
      - Encoder: extract spatial features from each radar frame (Conv2d).
      - Temporal aggregator: learn evolution across time (GRU).
      - Decoder: map aggregated features back to a 1-channel precipitation map.
    Input shape:  (B, T, C=1, H, W)
    Output shape: (B, 1, H, W)
    """

    def __init__(self, in_channels=1, enc_channels=32, gru_hidden=64, out_channels=1):
        super().__init__()

        # --- 1) Spatial encoder (small + fast) ---
        # Keeps HxW the same (padding=1 on 3x3)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, enc_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # --- 2) Temporal aggregator (GRU) ---
        # We’ll feed per-pixel feature vectors across time into a GRU.
        # Input size to GRU is enc_channels; it outputs gru_hidden features.
        # batch_first=False because we use (T, N, F) layout for RNNs below.
        self.gru = nn.GRU(input_size=enc_channels, hidden_size=gru_hidden, batch_first=False)

        # --- 3) Decoder back to 1 map ---
        self.decoder = nn.Sequential(
            nn.Conv2d(gru_hidden, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        x: (B, T, C=1, H, W)
        Steps:
          - Encode each time step -> (B, Cenc, H, W)
          - Stack over T -> (T, B, Cenc, H, W)
          - Rearrange to per-pixel sequences -> (T, B*H*W, Cenc)
          - GRU over T -> last hidden -> (1, B*H*W, Hgru)
          - Reshape back to (B, Hgru, H, W)
          - Decode to (B, 1, H, W)
        """
        B, T, C, H, W = x.shape
        assert C == 1, f"Expected 1 input channel; got {C}"

        # Encode each frame independently
        encoded_per_t = []
        for t in range(T):
            # x[:, t] -> (B, 1, H, W)
            feat = self.encoder(x[:, t])  # (B, Cenc, H, W)
            encoded_per_t.append(feat)

        # Stack along time -> (T, B, Cenc, H, W)
        enc = torch.stack(encoded_per_t, dim=0)

        # Prepare for GRU: we want sequences over time per spatial location.
        # Move (T, B, Cenc, H, W) -> (T, B*H*W, Cenc)
        T_, B_, Cenc, H_, W_ = enc.shape
        enc = enc.permute(0, 1, 3, 4, 2).contiguous()   # (T, B, H, W, Cenc)
        enc = enc.view(T_, B_ * H_ * W_, Cenc)          # (T, B*H*W, Cenc)

        # GRU over time; we only need the last hidden state as summary
        # h_n: (num_layers=1, N=B*H*W, Hgru)
        _, h_n = self.gru(enc)                          # h_n shape: (1, B*H*W, Hgru)
        h_last = h_n[0]                                 # (B*H*W, Hgru)

        # Reshape back to (B, Hgru, H, W)
        Hgru = h_last.shape[-1]
        agg = h_last.view(B_, H_, W_, Hgru)             # (B, H, W, Hgru)
        agg = agg.permute(0, 3, 1, 2).contiguous()      # (B, Hgru, H, W)

        # Decode to 1 channel map
        out = self.decoder(agg)                         # (B, 1, H, W)
        return out


# ---- Dummy dataset to verify the full training loop works without real data ----
class DummyRadarDataset(Dataset):
    """
    Generates random radar sequences and a random target.
    Purpose: let you verify the model & training loop end-to-end.
    """
    def __init__(self, num_samples=64, time_steps=6, H=64, W=64):
        super().__init__()
        self.num_samples = num_samples
        self.T = time_steps
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input sequence: (T, C=1, H, W)
        x = torch.randn(self.T, 1, self.H, self.W)
        # Target (next-step precipitation map or any supervision you choose); here we just learn identity-ish mapping
        y = torch.randn(1, self.H, self.W)
        return x, y


def shape_test():
    print(">>> Running shape test...")
    model = MiniMetNet()
    dummy = torch.randn(2, 6, 1, 64, 64)  # (B=2, T=6, C=1, H=64, W=64)
    out = model(dummy)
    print("Input shape :", tuple(dummy.shape))
    print("Output shape:", tuple(out.shape))
    assert out.shape == (2, 1, 64, 64), "Output shape is not (2, 1, 64, 64)."
    print("✓ Shape test passed.\n")


def quick_train_demo(device="cuda"):
    print(">>> Starting quick dummy training demo...")
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model / Loss / Optimizer
    model = MiniMetNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Data
    dataset = DummyRadarDataset(num_samples=128, time_steps=6, H=64, W=64)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Train for a couple of epochs just to verify everything runs
    epochs = 2
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            # x: (B, T, 1, H, W) -> move to device
            x = x.to(device)
            y = y.to(device)

            # Forward
            pred = model(x)          # (B, 1, H, W)
            loss = criterion(pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / len(loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Save a checkpoint each epoch
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
        print(f"Saved checkpoint: checkpoint_epoch_{epoch}.pt")

    print("✓ Dummy training completed.\n")


if __name__ == "__main__":
    # 1) Sanity/shape test (no GPU required)
    shape_test()

    # 2) Quick dummy training (uses GPU if available)
    quick_train_demo(device="cuda")
