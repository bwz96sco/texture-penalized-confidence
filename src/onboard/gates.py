"""Gate scores for a batch of tiles: the controls of BASELINES, computed on the main model's outputs.

conf-gate-0.70/v1: mean over pixels of the main model's top-class probability.
selective-ood/v1:  minus the mean pixel entropy of the five members' averaged softmax.
range-check/v1:    number of activations (all pixels, all layers) outside per-layer bounds
                   from source development data, or non-finite.
dup-check/v1:      run twice and compare; evaluated in the fault code.
"""
import numpy as np
import torch

from onboard.model import CLOUD
from onboard import scoring


@torch.no_grad()
def tile_scores(main, others, x_tiles, bounds=None, hook=None, keep_logits=False):
    """Scores and outputs for (N, 64, 64, 4) tiles. `main` may carry a fault (hook or weights)."""
    acts = []

    def record(i, a):
        if hook is not None:
            a = hook(i, a)
        acts.append(a)
        return a

    logits = scoring.forward_tiles(main, x_tiles, hook=record)          # (N, P, 3)
    p_main = torch.softmax(logits, dim=-1)
    pred = logits.argmax(-1)
    out = {
        "pred_cloud_fraction": (pred == CLOUD).float().mean(1).numpy(),
        "conf": p_main.max(-1).values.mean(1).numpy(),
        "mean_softmax": p_main.mean(1).numpy(),
        "penultimate_mean": acts[1].reshape(len(x_tiles), -1, acts[1].shape[-1]).mean(1).numpy(),
    }
    p_all = p_main.clone()
    for m in others:
        p_all += torch.softmax(scoring.forward_tiles(m, x_tiles), dim=-1)
    p_avg = p_all / (1 + len(others))
    ent = -(p_avg * torch.log(p_avg.clamp_min(1e-12))).sum(-1)
    out["ens"] = (-ent.mean(1)).numpy()
    if bounds is not None:
        v = torch.zeros(len(x_tiles), dtype=torch.int64)
        for a, (lo, hi) in zip(acts, bounds):
            a = a.reshape(len(x_tiles), -1)
            bad = ~torch.isfinite(a) | (a < lo) | (a > hi)
            v += bad.sum(1)
        out["violations"] = v.numpy()
    if keep_logits:
        out["logits"] = logits
    return out
