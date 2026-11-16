"""Single-bit flips in float32 activations (transient) and weights (persistent)."""
import copy

import numpy as np
import torch


def flip(value: torch.Tensor, flat_index: int, bit: int) -> torch.Tensor:
    """Return a copy of a float32 tensor with one bit of one element flipped."""
    out = value.detach().clone().contiguous()
    as_int = out.view(-1).view(torch.int32)
    as_int[flat_index] ^= np.int32(np.uint32(1 << bit).view(np.int32))
    return out


def transient_hook(layer: int, flat_index: int, bit: int):
    """Forward hook flipping one bit of one activation of `layer`'s output for a tile."""
    def hook(i, a):
        return flip(a, flat_index, bit) if i == layer else a
    return hook


def persistent_model(model, layer: int, flat_index: int, bit: int):
    """A copy of `model` with one bit of one weight of `layer` flipped."""
    m = copy.deepcopy(model)
    w = m.layers[layer].weight
    with torch.no_grad():
        w.copy_(flip(w, flat_index, bit))
    return m
