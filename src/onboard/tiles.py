"""64 x 64-pixel tiles with edge remainders dropped."""
import numpy as np

TILE = 64


def grid(h: int, w: int):
    """Top-left corners of full tiles, row-major."""
    return [(r, c) for r in range(0, h - TILE + 1, TILE) for c in range(0, w - TILE + 1, TILE)]


def cut(a: np.ndarray, corners) -> np.ndarray:
    """Stack tiles of an (H, W, ...) array: (N, TILE, TILE, ...)."""
    return np.stack([a[r:r + TILE, c:c + TILE] for r, c in corners])
