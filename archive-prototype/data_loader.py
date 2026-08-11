import os
import wradlib as wrl
import torch
import numpy as np


def load_radolan_tensor(path, center_crop=128):
    data, attrs = wrl.io.read_radolan_composite(path)
    data = np.ma.filled(data, fill_value=0).astype(np.float32)

    if center_crop:
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        data = data[cy - center_crop // 2:cy + center_crop // 2,
               cx - center_crop // 2:cx + center_crop // 2]
    data /= (data.max() + 1e-6)  # normalize
    return torch.tensor(data).unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]


def load_sequence(filepaths):
    frames = [load_radolan_tensor(p) for p in filepaths]
    return torch.cat(frames, dim=1)  # shape [1, 6, H, W]
