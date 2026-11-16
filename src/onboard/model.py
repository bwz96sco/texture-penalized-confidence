"""Per-pixel MLP on four bands, with standardisation inside the model (float32)."""
import torch
from torch import nn

N_CLASSES = 3  # water, land, cloud
CLOUD = 2


class PixelMLP(nn.Module):
    def __init__(self, n_bands: int = 4, hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_bands))
        self.register_buffer("std", torch.ones(n_bands))
        self.layers = nn.ModuleList([nn.Linear(n_bands, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, N_CLASSES)])

    def forward(self, x: torch.Tensor, hook=None) -> torch.Tensor:
        """Logits for (N, n_bands) pixels. `hook(i, a)` sees and may replace each layer's output."""
        a = (x - self.mean) / self.std
        for i, layer in enumerate(self.layers):
            a = layer(a)
            if i < len(self.layers) - 1:
                a = torch.relu(a)
            if hook is not None:
                a = hook(i, a)
        return a
