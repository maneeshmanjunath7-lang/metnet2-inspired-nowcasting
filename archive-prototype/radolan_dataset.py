# radolan_dataset.py
# Starter dataset for RADOLAN (e.g., RW) sequences with sliding-window sampling.
# Requires: pip install wradlib numpy torch pillow

import os
import re
import io
import tarfile
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset

# Try import wradlib
try:
    import wradlib as wrl
    HAVE_WRADLIB = True
except Exception:
    HAVE_WRADLIB = False


def _natural_key(s: str):
    # Sort like humans (…_0001, …_0002, …_0010)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_radolan_members(sources: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Recursively collect files from directories and members from .tar archives.
    Returns a time-sorted list of (path, member_name) pairs:
      - For normal files on disk: (full_path, None)
      - For files inside a .tar:  (tar_path, member_name)
    """
    items = []

    def add_file(path: str):
        if os.path.isfile(path):
            items.append((path, None))

    for src in sources:
        if os.path.isdir(src):
            # RECURSIVE walk
            for root, dirs, files in os.walk(src):
                for fn in files:
                    add_file(os.path.join(root, fn))
        elif src.lower().endswith(".tar"):
            with tarfile.open(src, "r") as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        items.append((src, m.name))
        else:
            add_file(src)

    # Natural sort by "path::member"
    items.sort(key=lambda x: _natural_key(f"{x[0]}::{x[1] or ''}"))
    return items



def _read_radolan_array(path: str, member: Optional[str]) -> np.ndarray:
    """
    Load a single RADOLAN file (RW etc.) and return data as float32 (mm/h).
    Supports reading from inside a .tar without extracting.
    """
    if not HAVE_WRADLIB:
        raise ImportError(
            "wradlib is required. Install with: pip install wradlib"
        )

    if member is None and not path.lower().endswith(".tar"):
        # Normal file on disk
        data, attrs = wrl.io.radolan.read_radolan_composite(path)
    elif path.lower().endswith(".tar"):
        # File inside a tar
        with tarfile.open(path, "r") as tf:
            ti = tf.getmember(member)
            fb = tf.extractfile(ti)
            raw = fb.read()
        bio = io.BytesIO(raw)
        data, attrs = wrl.io.radolan.read_radolan_composite(bio)
    else:
        # Fallback
        data, attrs = wrl.io.radolan.read_radolan_composite(path)

    # Convert to float32 mm/h; mask/clean invalids
    arr = np.array(data, dtype=np.float32)
    # RADOLAN uses -9999 etc. for missing; also attrs may contain 'nodataflag'
    arr[arr < 0] = 0.0
    return arr  # mm/h for RW


def _center_crop_or_resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Center-crop or pad/resize to (H,W) using PIL for robustness.
    arr: (H, W) float32
    """
    H, W = arr.shape
    target_h, target_w = size

    if (H, W) == (target_h, target_w):
        return arr

    # Use PIL to resize with bilinear (good enough for starter)
    im = Image.fromarray(arr)
    im = im.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(im, dtype=np.float32)


def _normalize_mm_per_h(arr: np.ndarray, mode: str = "log1p") -> np.ndarray:
    """
    Normalize precipitation rates.
    mode='log1p': x' = log1p(x) / log1p( max_mmph )
    We use max_mmph=100 as a tame cap for starter.
    """
    if mode == "none":
        return arr
    if mode == "log1p":
        max_mmph = 100.0
        arr = np.clip(arr, 0, max_mmph)
        arr = np.log1p(arr) / np.log1p(max_mmph)
        return arr
    raise ValueError(f"Unknown normalization: {mode}")


class RadolanSequenceDataset(Dataset):
    """
    RADOLAN sequence dataset (folders or .tar files).
    Produces sliding windows of T past frames -> 1 target frame.

    Returns:
      x: float32 tensor (T, 1, H, W)
      y: float32 tensor (1, H, W)

    Args:
      sources: list of directories / .tar files / single files (mixed allowed)
      time_steps: number of input frames (T)
      target_offset: how many steps ahead as target (usually 1)
      crop_size: (H,W) for model; images are resized to this
      stride: step between window starts
      cache_dir: if given, cached .npy files are written/read for speed
      normalization: 'log1p' or 'none'
      product_hint: optional string for filtering (e.g., 'RW'), simple contains match
    """

    def __init__(
        self,
        sources: List[str],
        time_steps: int = 6,
        target_offset: int = 1,
        crop_size: Tuple[int, int] = (64, 64),
        stride: int = 1,
        cache_dir: Optional[str] = None,
        normalization: str = "log1p",
        product_hint: Optional[str] = None,
    ):
        super().__init__()
        self.sources = sources
        self.T = time_steps
        self.target_offset = target_offset
        self.size = crop_size
        self.stride = stride
        self.cache_dir = cache_dir
        self.norm = normalization
        self.product_hint = product_hint

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 1) Build sorted list of (path, member)
        all_items = list_radolan_members(self.sources)

        # 2) Optional filtering by product name
        if self.product_hint:
            all_items = [
                it for it in all_items
                if (it[1] or it[0]).upper().find(self.product_hint.upper()) != -1
            ]

        if len(all_items) < (self.T + self.target_offset):
            raise ValueError("Not enough files to form a single sequence.")

        # 3) Build sliding windows (start indices)
        self.frames = all_items
        self.indices = []
        for start in range(0, len(self.frames) - (self.T + self.target_offset) + 1, self.stride):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def _cache_key(self, path: str, member: Optional[str]) -> str:
        base = path.replace(os.sep, "_")
        if member:
            base += f"__{member.replace(os.sep,'_')}"
        return os.path.join(self.cache_dir, base + ".npy")

    def _load_single_frame(self, path: str, member: Optional[str]) -> np.ndarray:
        # Try cache
        if self.cache_dir:
            ck = self._cache_key(path, member)
            if os.path.exists(ck):
                return np.load(ck)

        arr = _read_radolan_array(path, member)  # (H, W) float32 mm/h
        arr = _center_crop_or_resize(arr, self.size)  # (H,W)
        arr = _normalize_mm_per_h(arr, self.norm)     # normalized float32

        if self.cache_dir:
            np.save(ck, arr)
        return arr

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        # x-frames: [start .. start+T-1], target at (start+T-1 + target_offset)
        seq_items = self.frames[start : start + self.T]
        tgt_item = self.frames[start + self.T - 1 + self.target_offset]

        # Load and stack inputs
        x_list = []
        for (path, member) in seq_items:
            arr = self._load_single_frame(path, member)  # (H,W)
            x_list.append(arr[None, ...])  # add channel dim -> (1,H,W)

        x = np.stack(x_list, axis=0).astype(np.float32)  # (T,1,H,W)

        # Load target
        y_arr = self._load_single_frame(tgt_item[0], tgt_item[1])  # (H,W)
        y = y_arr[None, ...].astype(np.float32)  # (1,H,W)

        # To torch
        x = torch.from_numpy(x)  # (T,1,H,W)
        y = torch.from_numpy(y)  # (1,H,W)
        return x, y
