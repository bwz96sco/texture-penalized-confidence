"""Launch check: clean runs repeat bit for bit; an exponent-bit flip changes the output.

Usage: python -I check_injector.py SPARCS_DERIVED_DIR MODELS_DIR
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from onboard import faults, scoring, tiles

src, models = sys.argv[1], sys.argv[2]
torch.set_num_threads(8)
split = json.load(open(f"{models}/split.json"))
main = scoring.load_member(f"{models}/member0.pt", C.HIDDEN)
n = split["dev"][0]
x = np.load(f"{src}/{n}_toa.npy")
xt = tiles.cut(x, tiles.grid(*x.shape[:2]))[:8]
a = scoring.forward_tiles(main, xt); b = scoring.forward_tiles(main, xt)
identical = torch.equal(a.view(torch.int32), b.view(torch.int32))
changed = []
for layer in range(3):
    for t in range(len(xt)):
        out = scoring.forward_tiles(main, xt[t:t + 1], hook=faults.transient_hook(layer, 5, 30))
        changed.append(not torch.equal(out, a[t:t + 1]))
pm = faults.persistent_model(main, 1, 7, 30)
pchanged = not torch.equal(scoring.forward_tiles(pm, xt), a)
untouched = torch.equal(scoring.forward_tiles(main, xt), a)
print(f"clean runs bit-identical: {identical}; transient bit-30 flips changing output: {sum(changed)}/{len(changed)}; "
      f"persistent bit-30 flip changes output: {pchanged}; original model untouched: {untouched}")
