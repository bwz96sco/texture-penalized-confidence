"""Per-tile outputs, gate scores and label-derived correctness."""
import numpy as np
import torch

from onboard import tiles
from onboard.model import CLOUD

TILE_PIX = tiles.TILE * tiles.TILE


def load_member(path: str, hidden: int = 64):
    from onboard.model import PixelMLP
    m = PixelMLP(hidden=hidden)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def forward_tiles(model, x_tiles: np.ndarray, hook=None):
    """x_tiles (N, 64, 64, 4) -> logits (N, 4096, 3) float32."""
    n = len(x_tiles)
    x = torch.from_numpy(np.ascontiguousarray(x_tiles.reshape(n * TILE_PIX, 4)))
    return model(x, hook=hook).reshape(n, TILE_PIX, -1)


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def implied_downlink(pred_cloud_fraction: np.ndarray, threshold: float) -> np.ndarray:
    return pred_cloud_fraction < threshold


def label_tiles(label_tiles: np.ndarray, cloud_code: int, threshold: float):
    """Label cloud fraction per tile and the label's implied downlink decision."""
    frac = (label_tiles.reshape(len(label_tiles), -1) == cloud_code).mean(axis=1)
    return frac, frac < threshold
