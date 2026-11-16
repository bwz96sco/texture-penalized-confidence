"""Post-freeze data-validity check: are the HYPSO-1 labels aligned with the calibrated cube?

Usage: python -I check_label_alignment.py HYPSO1_DERIVED HYPSO1_DOWNLOAD MODELS TEST_LIST RESERVED_LIST OUT_CSV
Compares the main model's per-pixel prediction with each inspectable capture's label array as
published (C-order reshape to 956 x 684) and under four alternative orientations. Reserved
captures are skipped. Sets no threshold and changes nothing in the frozen evaluation.
"""
import csv, sys

import numpy as np
import torch

from onboard import scoring

hypso_dir, download, models, test_list, reserved_list, out = sys.argv[1:7]
SHAPE = (956, 684)


def read_list(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


reserved = set(read_list(reserved_list))
captures = [c for c in sorted(read_list(test_list)) if c not in reserved]
assert len(captures) == 28
main = scoring.load_member(f"{models}/member0.pt")
orientations = {
    "as_published": lambda raw: raw.reshape(SHAPE),
    "flip_rows": lambda raw: raw.reshape(SHAPE)[::-1],
    "flip_cols": lambda raw: raw.reshape(SHAPE)[:, ::-1],
    "rot180": lambda raw: raw.reshape(SHAPE)[::-1, ::-1],
    "stored_transposed": lambda raw: raw.reshape(SHAPE[::-1]).T,
}
rows = []
for cap in captures:
    x = np.load(f"{hypso_dir}/{cap}_toa.npy")
    with torch.no_grad():
        pred = main(torch.from_numpy(x.reshape(-1, 4))).argmax(-1).numpy().reshape(SHAPE)
    name = cap.split("-", 1)[1]
    raw = np.fromfile(f"{download}/{cap}/{name}-slc-labels.bin", dtype=np.uint8)
    row = {"capture": cap}
    for k, f in orientations.items():
        lab = f(raw)
        row[f"{k}_3class"] = float((pred == lab).mean())
        row[f"{k}_cloud"] = float(((pred == 2) == (lab == 2)).mean())
    rows.append(row)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
for k in orientations:
    a3 = [r[f"{k}_3class"] for r in rows]; ac = [r[f"{k}_cloud"] for r in rows]
    best = sum(r[f"{k}_3class"] >= max(r[f"{o}_3class"] for o in orientations) for r in rows)
    print(f"{k:18s} 3-class mean {np.mean(a3):.4f} (min {min(a3):.4f})  cloud mean {np.mean(ac):.4f}  best in {best}/28")
