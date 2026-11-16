"""Launch check: the main model's tile decision accuracy on SPARCS development subscenes.

Usage: python -I check_model_floor.py SPARCS_DERIVED_DIR MODELS_DIR
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from onboard import scoring, tiles
from onboard.model import CLOUD
from onboard.sparcs import CLOUD_CODE

src, models = sys.argv[1], sys.argv[2]
split = json.load(open(f"{models}/split.json"))
main = scoring.load_member(f"{models}/member0.pt", C.HIDDEN)
correct = []
for n in split["dev"]:
    x = np.load(f"{src}/{n}_toa.npy"); lab = np.load(f"{src}/{n}_labels.npy")
    corners = tiles.grid(*lab.shape)
    p = scoring.softmax(scoring.forward_tiles(main, tiles.cut(x, corners))).numpy()
    pred_frac = (p.argmax(-1) == CLOUD).mean(axis=1)
    _, lab_down = scoring.label_tiles(tiles.cut(lab, corners), CLOUD_CODE, C.CLOUD_DECISION)
    correct.append(scoring.implied_downlink(pred_frac, C.CLOUD_DECISION) == lab_down)
c = np.concatenate(correct)
print(f"dev subscenes {len(split['dev'])}, tiles {len(c)}, decision accuracy {c.mean():.4f}, floor {C.MODEL_FLOOR}, pass {c.mean() >= C.MODEL_FLOOR}")
