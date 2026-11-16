"""Scored run: the frozen stand-in on CloudSEN12+'s 535 high-quality 509-pixel validation patches.

Usage: python -I run_room.py MLSTAC_FILE METADATA_CSV MODELS PREMISE_EVIDENCE TASK_EVIDENCE
Requires TASK_EVIDENCE/download.json with passes = true and the padding side found there. Writes
room-tiles.csv and results.json to TASK_EVIDENCE. Runs once (contract, Stop conditions).
"""
import csv, hashlib, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlstac
from onboard import gates, metrics, scoring, tiles

path, meta_csv, models, premise_evid, evid = sys.argv[1:6]
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
COVERAGE, CLOUD, SAT_SHARE, SAT_DN, SZA_MAX = 0.8, 0.5, 0.1, 65535, 60.1
DRAWS, SEED = 2000, 7

dl = json.load(open(f"{evid}/download.json"))
assert dl["passes"], "download checks did not pass"
side = tuple(dl["first_patch"]["padding_side_row_col_start"])
pf = json.load(open(f"{premise_evid}/freeze.json"))
for k, h in pf["model"]["member_sha256"].items():
    assert hashlib.sha256(open(f"{models}/member{k}.pt", "rb").read()).hexdigest() == h, f"member {k} changed"
members = [scoring.load_member(f"{models}/member{k}.pt", 64) for k in range(5)]
bounds = [tuple(b) for b in pf["range_bounds"]]

meta = {r["datapoint_id"]: r for r in csv.DictReader(open(meta_csv))
        if r["split"] == "validation" and r["proj_shape"] == "509" and r["label_type"] == "high"}
_, length, table = mlstac.check(path, list(meta.values()))
assert set(table) == set(meta) and len(meta) == 535

rows = []
for pid in sorted(meta):
    a, info = mlstac.read_patch(path, length, *table[pid], bands=[1, 2, 3, 4, 14])
    img, lab = a[:4], a[4]
    assert info["count"] == 15 and mlstac.padding_is_zero(img, lab, side), pid
    img, lab = mlstac.crop(img, side), mlstac.crop(lab, side)
    assert set(np.unique(lab).tolist()) <= {0, 1, 2, 3}, pid
    sat = (img >= SAT_DN).any(0)
    x = np.moveaxis(img, 0, -1).astype(np.float32) * np.float32(1e-4)
    corners = tiles.grid(*x.shape[:2])
    xt = tiles.cut(x, corners)
    s = gates.tile_scores(members[0], members[1:], xt, bounds)
    lt = tiles.cut(lab, corners).reshape(len(corners), -1)
    st = tiles.cut(sat, corners).reshape(len(corners), -1)
    for j in range(len(corners)):
        rows.append({"patch": pid, "roi": meta[pid]["roi_id"], "tile": j,
                     "sza": float(meta[pid]["s2_mean_solar_zenith_angle"]),
                     "conf": float(s["conf"][j]), "ens": float(s["ens"][j]),
                     "pred_cloud_frac": float(s["pred_cloud_fraction"][j]),
                     "thick_frac": float((lt[j] == 1).mean()), "thin_frac": float((lt[j] == 2).mean()),
                     "sat_frac": float(st[j].mean())})
assert len(rows) == 535 * 49

cols = list(rows[0])
with open(f"{evid}/room-tiles.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for r in rows:
        w.writerow([repr(r[k]) if isinstance(r[k], float) else r[k] for k in cols])

# --- measurement -------------------------------------------------------------------------------------
t = {k: np.array([r[k] for r in rows]) for k in cols}
pred = t["pred_cloud_frac"] >= CLOUD
sat = (t["sat_frac"] > SAT_SHARE) & ~pred
base = t["ens"] - 1e6 * sat
rois = sorted(set(t["roi"]))
groups = {r: np.flatnonzero(t["roi"] == r) for r in rois}


def label_cloud(conv):
    return (t["thick_frac"] + (t["thin_frac"] if conv == "B" else 0)) >= CLOUD


def summary(conv):
    correct = pred == label_cloud(conv)
    k = math.ceil(COVERAGE * len(base))
    acc = np.zeros(len(base), bool); acc[np.argsort(-base, kind="stable")[:k]] = True
    aw = acc & ~correct
    risk = metrics.selective_risk(base, correct, COVERAGE)
    orc = metrics.oracle_risk(correct, COVERAGE)
    draws = metrics.cluster_draws(len(rois), DRAWS, SEED)
    boot = []
    for d in draws:
        idx = np.concatenate([groups[rois[i]] for i in d])
        boot.append(metrics.selective_risk(base[idx], correct[idx], COVERAGE) - metrics.oracle_risk(correct[idx], COVERAGE))
    lo, hi = metrics.percentile_interval(boot)
    fc = aw & pred
    out = {"accuracy": float(correct.mean()), "baseline_risk": risk, "oracle_risk": orc,
           "room_points": 100 * (risk - orc), "room_ci": [100 * lo, 100 * hi],
           "conf_gate_risk": metrics.selective_risk(t["conf"], correct, COVERAGE),
           "n_accepted": int(k), "n_accepted_wrong": int(aw.sum()),
           "accepted_wrong_false_cloud": int(fc.sum()), "accepted_wrong_missed_cloud": int((aw & ~pred).sum()),
           "false_cloud_share": float(fc.sum() / max(aw.sum(), 1)),
           "accepted_wrong_sza_above_60_1": int((aw & (t["sza"] > SZA_MAX)).sum()),
           "false_cloud_thin_share_at_least_half": int((fc & (t["thin_frac"] >= 0.5)).sum())}
    return out


res = {"n_patches": 535, "n_rois": len(rois), "n_tiles": len(rows), "padding_side": list(side),
       "saturated_pixel_share": float(t["sat_frac"].mean()), "tiles_with_saturation_rule": int(sat.sum()),
       "model_cloud_tile_share": float(pred.mean()), "tiles_sza_above_60_1": int((t["sza"] > SZA_MAX).sum()),
       "A": summary("A"), "B": summary("B")}
res["room_A_minus_B"] = res["A"]["room_points"] - res["B"]["room_points"]
A = res["A"]
if A["room_points"] >= 3 and A["false_cloud_share"] >= 0.5:
    res["outcome"] = "open"
elif A["room_points"] >= 3:
    res["outcome"] = "narrow"
else:
    res["outcome"] = "closed"
json.dump(res, open(f"{evid}/results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
