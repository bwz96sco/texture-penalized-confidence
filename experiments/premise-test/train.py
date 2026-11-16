"""Train the five per-pixel members on the SPARCS training subscenes.

Usage: python -I train.py SPARCS_DERIVED_DIR OUT_DIR
Writes OUT_DIR/split.json, OUT_DIR/member{k}.pt and OUT_DIR/train-log.json.
Only training-split data is read.
"""
import glob, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from onboard.model import PixelMLP
from onboard.sparcs import MERGE

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)

names = sorted(os.path.basename(p)[:-len("_toa.npy")] for p in glob.glob(f"{src}/*_toa.npy"))
assert len(names) == 78, len(names)
order = np.random.default_rng(C.SPLIT_SEED).permutation(len(names))
split = {"train": sorted(names[i] for i in order[:C.N_TRAIN]), "dev": sorted(names[i] for i in order[C.N_TRAIN:])}
json.dump(split, open(f"{out}/split.json", "w"), indent=1)

xs, ys = [], []
for n in split["train"]:
    x = np.load(f"{src}/{n}_toa.npy").reshape(-1, 4)
    y = MERGE[np.load(f"{src}/{n}_labels.npy").reshape(-1)]
    keep = (y != 255) & np.isfinite(x).all(axis=1)
    xs.append(x[keep]); ys.append(y[keep])
X = torch.from_numpy(np.concatenate(xs)); Y = torch.from_numpy(np.concatenate(ys).astype(np.int64))
by_class = [torch.nonzero(Y == c).squeeze(1) for c in range(3)]
mean, std = X.mean(0), X.std(0)
log = {"train_pixels": int(len(Y)), "class_counts": [int(len(i)) for i in by_class],
       "mean": mean.tolist(), "std": std.tolist(), "members": {}}

for k in C.MEMBERS:
    torch.manual_seed(k)
    g = torch.Generator().manual_seed(1000 + k)
    m = PixelMLP(hidden=C.HIDDEN)
    m.mean.copy_(mean); m.std.copy_(std)
    opt = torch.optim.Adam(m.parameters(), lr=C.LR)
    hist, t0 = [], time.time()
    for ep in range(C.EPOCHS):
        idx = torch.cat([i[torch.randint(len(i), (C.PER_CLASS_PER_EPOCH,), generator=g)] for i in by_class])
        idx = idx[torch.randperm(len(idx), generator=g)]
        tot = 0.0
        for b in range(0, len(idx), C.BATCH):
            j = idx[b:b + C.BATCH]
            loss = torch.nn.functional.cross_entropy(m(X[j]), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        hist.append(tot / len(idx))
    with torch.no_grad():
        j = torch.randint(len(Y), (500_000,), generator=g)
        acc = (m(X[j]).argmax(1) == Y[j]).float().mean().item()
    torch.save(m.state_dict(), f"{out}/member{k}.pt")
    log["members"][k] = {"loss_per_epoch": hist, "train_pixel_accuracy_sample": acc, "seconds": time.time() - t0}
    print(k, [round(h, 4) for h in hist], round(acc, 4), round(time.time() - t0, 1), flush=True)
json.dump(log, open(f"{out}/train-log.json", "w"), indent=1)
