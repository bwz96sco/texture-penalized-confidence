"""Write freeze.json from SPARCS data only: bounds, thresholds, checksums and configuration.

Usage: python -I make_freeze.py SPARCS_DERIVED_DIR MODELS_DIR HYPSO1_CAL_CSV HYPSO1_CHANNELS_JSON OUT_JSON
"""
import hashlib, json, os, subprocess, sys
from importlib.metadata import version

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from onboard import gates, scoring, tiles
from onboard.hypso1 import COEFF_TYPE, LANDSAT_BANDS

src, models, cal_csv, channels_json, out = sys.argv[1:6]
torch.set_num_threads(8)
split = json.load(open(f"{models}/split.json"))
members = [scoring.load_member(f"{models}/member{k}.pt", C.HIDDEN) for k in C.MEMBERS]
main, others = members[0], members[1:]

lo = [np.inf] * 3; hi = [-np.inf] * 3
conf, ens = [], []
for n in split["dev"]:
    x = np.load(f"{src}/{n}_toa.npy")
    xt = tiles.cut(x, tiles.grid(*x.shape[:2]))
    acts = []
    with torch.no_grad():
        scoring.forward_tiles(main, xt, hook=lambda i, a: acts.append(a) or a)
    for i, a in enumerate(acts):
        lo[i] = min(lo[i], float(a.min())); hi[i] = max(hi[i], float(a.max()))
    s = gates.tile_scores(main, others, xt)
    conf.append(s["conf"]); ens.append(s["ens"])
conf, ens = np.concatenate(conf), np.concatenate(ens)

sha = {k: hashlib.sha256(open(f"{models}/member{k}.pt", "rb").read()).hexdigest() for k in C.MEMBERS}
freeze = {
    "contract_revision": 1,
    "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    "split": split,
    "model": {"architecture": "PixelMLP 4-64-64-3, ReLU, standardisation inside", "member_sha256": sha},
    "range_bounds": [[lo[i], hi[i]] for i in range(3)],
    "fault_thresholds": {"conf": float(np.quantile(conf, C.CLEAN_FALSE_REJECT)),
                         "ens": float(np.quantile(ens, C.CLEAN_FALSE_REJECT))},
    "source_dev_tiles": int(len(conf)),
    "hypso": {"version": version("hypso"), "hypso1_calibration": version("hypso1-calibration"),
              "coeff_type": COEFF_TYPE, "toa": "generate_l1d_cube(use_direct_georef=True)",
              "landsat_bands_nm": LANDSAT_BANDS, "channels": json.load(open(channels_json))},
    "torch": torch.__version__,
    "nonfinite_rule": "a non-finite gate score counts as reject; argmax follows PyTorch",
    "tie_rule": "selective risk accepts the ceil(coverage * N) best scores; stable order breaks ties",
}
json.dump(freeze, open(out, "w"), indent=1)
print(json.dumps({k: freeze[k] for k in ("range_bounds", "fault_thresholds", "source_dev_tiles")}, indent=1))
