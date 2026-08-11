import os, glob, torch
from radolan_dataset import RadolanSequenceDataset

BASE = r"C:\Maneesh\Mission Germany\TUM Aerospace\Studies\Semester Thesis\Weather Forecasting\Project Code\radolan_data"

# Recursively collect directories and .tar files
SOURCES = []
for p in glob.glob(BASE + r"\**", recursive=True):
    if os.path.isdir(p) or p.lower().endswith(".tar"):
        SOURCES.append(p)
# also include BASE itself if it's a flat folder of files
if BASE not in SOURCES:
    SOURCES.append(BASE)

print(f"Collected {len(SOURCES)} source locations.")
for s in SOURCES[:5]:
    print("  -", s)

# Build dataset
ds = RadolanSequenceDataset(
    sources=SOURCES,
    time_steps=2,         # smaller for the first check
    target_offset=1,
    crop_size=(64, 64),
    stride=1,
    cache_dir=os.path.join(BASE, "_cache"),
    normalization="log1p",
    product_hint=None,    # <-- remove the filter for now
)


print(f"\nDataset windows: {len(ds)}")
# Peek one sample
x, y = ds[0]
print("x shape (T,1,H,W):", tuple(x.shape))
print("y shape (1,H,W):  ", tuple(y.shape))
print("x min/max:", float(x.min()), float(x.max()))
print("y min/max:", float(y.min()), float(y.max()))
