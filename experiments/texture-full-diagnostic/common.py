"""Fixed visible-texture and small spatial checks; no target fitting."""
import csv
import json
import math
import platform
import resource
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED, THREADS, COVERAGE, DRAWS = 20261015, 8, 0.8, 2000
BATCH, EPOCHS = 64, 20


def setup():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def peak_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def rotation_table():
    raw = np.arange(256, dtype=np.uint16)
    canonical = np.min(np.stack([((raw >> k) | (raw << (8 - k))) & 255
                                 for k in range(8)]), axis=0)
    keys = np.unique(canonical)
    assert len(keys) == 36
    return np.searchsorted(keys, canonical).astype(np.uint8)


LBP_BINS = rotation_table()


def texture(rgb):
    """(N,64,64,3) -> circular LBP histograms and equalized-red Sobel means."""
    rgb = np.asarray(rgb, dtype=np.float64)
    assert rgb.ndim == 4 and rgb.shape[1:] == (64, 64, 3)
    assert np.isfinite(rgb).all()
    gray = rgb.mean(axis=-1)
    center = gray[:, 3:-3, 3:-3]
    code = np.zeros(center.shape, dtype=np.uint8)
    for k in range(8):
        dy = round(3 * math.sin(2 * math.pi * k / 8), 12)
        dx = round(3 * math.cos(2 * math.pi * k / 8), 12)
        y, x = math.floor(3 + dy), math.floor(3 + dx)
        fy, fx = 3 + dy - y, 3 + dx - x
        near = (1 - fy) * (1 - fx) * gray[:, y:y + 58, x:x + 58]
        if fx:
            near = near + (1 - fy) * fx * gray[:, y:y + 58, x + 1:x + 59]
        if fy:
            near = near + fy * (1 - fx) * gray[:, y + 1:y + 59, x:x + 58]
        if fx and fy:
            near = near + fy * fx * gray[:, y + 1:y + 59, x + 1:x + 59]
        code |= (near >= center).astype(np.uint8) << k
    offsets = 36 * np.arange(len(rgb), dtype=np.int64)[:, None, None]
    hist = np.bincount((LBP_BINS[code] + offsets).ravel(),
                       minlength=36 * len(rgb)).reshape(len(rgb), 36) / (58 * 58)

    equal = np.zeros(gray.shape, dtype=np.float64)
    for i, red in enumerate(rgb[..., 2]):
        lo, hi = red.min(), red.max()
        if hi == lo:
            continue
        q = np.minimum(((red - lo) / (hi - lo) * 256).astype(np.int64), 255)
        counts = np.bincount(q.ravel(), minlength=256)
        cdf = counts.cumsum()
        first = cdf[counts > 0][0]
        equal[i] = (cdf[q] - first) / (red.size - first)
    gx = (equal[:, :-2, 2:] + 2 * equal[:, 1:-1, 2:] + equal[:, 2:, 2:]
          - equal[:, :-2, :-2] - 2 * equal[:, 1:-1, :-2] - equal[:, 2:, :-2]) / 8
    gy = (equal[:, 2:, :-2] + 2 * equal[:, 2:, 1:-1] + equal[:, 2:, 2:]
          - equal[:, :-2, :-2] - 2 * equal[:, :-2, 1:-1] - equal[:, :-2, 2:]) / 8
    return hist, np.hypot(gx, gy).mean(axis=(1, 2))


def distances(hist, templates):
    cloud, clear = np.asarray(templates["cloud"]), np.asarray(templates["clear"])
    dc = 0.5 * (((hist - cloud) ** 2) / (hist + cloud + 1e-12)).sum(1)
    dn = 0.5 * (((hist - clear) ** 2) / (hist + clear + 1e-12)).sum(1)
    q = (dc - dn) / (dc + dn + 1e-12)
    return dc, dn, q


class Spatial(nn.Module):
    """7x7 receptive field, within-tile context and three visible bands."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).float().reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).float().reshape(1, 3, 1, 1))
        self.net = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(3, 16, 3), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3), nn.ReLU(), nn.Conv2d(16, 2, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def load_spatial(path, mean, std):
    m = Spatial(mean, std)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


@torch.no_grad()
def spatial_scores(model, rgb):
    cloud, conf = [], []
    for i in range(0, len(rgb), BATCH):
        x = torch.from_numpy(np.ascontiguousarray(rgb[i:i + BATCH].transpose(0, 3, 1, 2))).float()
        logits = model(x)
        cloud.extend((logits.argmax(1) == 1).float().mean((1, 2)).tolist())
        conf.extend(logits.softmax(1).max(1).values.mean((1, 2)).tolist())
    return np.asarray(cloud), np.asarray(conf)
