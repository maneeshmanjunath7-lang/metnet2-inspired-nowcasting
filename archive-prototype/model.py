import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MetNet2Mini(nn.Module):
    def __init__(self, in_channels=6, num_bins=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, dilation=1),
            ConvBlock(32, 64, dilation=2),
            ConvBlock(64, 64, dilation=4)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_bins, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return F.softmax(x, dim=1)  # shape: [B, bins, H, W]
