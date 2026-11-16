"""Synthetic invariance/geometry checks before any target pass."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import LBP_BINS, Spatial, setup, texture
from analysis import accepted, inputs


def main():
    setup()
    rng = np.random.default_rng(55)
    rgb = rng.random((3, 64, 64, 3))
    h, g = texture(rgb)
    assert h.shape == (3, 36) and np.allclose(h.sum(1), 1)
    assert np.allclose(texture(2 * rgb + 0.3)[0], h)
    assert np.allclose(texture(np.rot90(rgb, axes=(1, 2)))[0], h)
    hc, gc = texture(np.ones((1, 64, 64, 3)))
    assert hc[0, LBP_BINS[255]] == 1 and np.count_nonzero(hc) == 1 and gc[0] == 0

    # Scalar bilinear reference detects swapped axes, missing radius and bad bit mapping.
    gray = rgb[0].mean(-1)
    slow = np.zeros(36)
    for y in range(3, 61):
        for x in range(3, 61):
            code = 0
            for k in range(8):
                yy = y + round(3 * math.sin(2 * math.pi * k / 8), 12)
                xx = x + round(3 * math.cos(2 * math.pi * k / 8), 12)
                yi, xi = math.floor(yy), math.floor(xx)
                fy, fx = yy - yi, xx - xi
                value = gray[yi, xi] * (1 - fy) * (1 - fx)
                if fx:
                    value += gray[yi, xi + 1] * (1 - fy) * fx
                if fy:
                    value += gray[yi + 1, xi] * fy * (1 - fx)
                if fy and fx:
                    value += gray[yi + 1, xi + 1] * fy * fx
                code |= int(value >= gray[y, x]) << k
            slow[LBP_BINS[code]] += 1
    assert np.allclose(slow / slow.sum(), h[0])

    model = Spatial([0, 0, 0], [1, 1, 1])
    x = torch.from_numpy(rgb[:1].transpose(0, 3, 1, 2)).float()
    logits = model(x)
    assert logits.shape == (1, 2, 64, 64)
    y = torch.zeros((1, 64, 64), dtype=torch.long)
    y[:, :3] = 255
    logits.retain_grad()
    torch.nn.functional.cross_entropy(logits, y, ignore_index=255).backward()
    assert logits.grad[:, :, :3].abs().max().item() == 0
    records = [{"pred_cloud_frac": float(i != 0), "spatial_cloud_frac": 0,
                "label_cloud_frac": 0, "thin_frac": 0, "sat_frac": 0.2 if i == 0 else 0,
                "ens": -0.1 * (i + 1), "conf": 0.9, "spatial_conf": 0.8,
                "lbp_q": 0.5, "lbp_dc": 0.3, "lbp_dn": 0.1, "gradient": 1}
               for i in range(5)]
    _, _, _, _, scores, _ = inputs(records, "A", {"gradient_threshold": 0.5})
    for name, score in scores.items():
        if name not in ("ensemble", "confidence"):
            assert int(np.argmin(score)) == 0, name
            assert not accepted(score)[0] and accepted(score).sum() == 4, name
    print(json.dumps({"synthetic_pass": True, "lbp_bins": 36,
                      "checks": ["scalar bilinear reference", "rotation and gain invariance",
                                 "constant texture", "CNN output geometry", "ignored-label gradient",
                                 "saturation outer priority", "global coverage rounding"]}))


if __name__ == "__main__":
    main()
