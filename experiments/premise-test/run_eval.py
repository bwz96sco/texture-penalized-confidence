"""Premise test after the freeze: source reference, shift-v0, faults and all metrics.

Usage: python -I run_eval.py SPARCS_DERIVED HYPSO1_DERIVED HYPSO1_DOWNLOAD MODELS FREEZE_JSON
                             TEST_LIST RESERVED_LIST EVIDENCE_DIR RUNS_DIR
Refuses to run unless FREEZE_JSON is committed unchanged. Writes the per-tile and per-fault
records and results.json to EVIDENCE_DIR, and probe features to RUNS_DIR.
"""
import csv, hashlib, json, math, os, subprocess, sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from onboard import faults, gates, metrics, scoring, tiles
from onboard.sparcs import CLOUD_CODE as SPARCS_CLOUD

HYPSO_CLOUD, HYPSO_SHAPE = 2, (956, 684)
(sparcs_dir, hypso_dir, download, models, freeze_path,
 test_list, reserved_list, evid, runs) = sys.argv[1:10]
os.makedirs(evid, exist_ok=True); os.makedirs(runs, exist_ok=True)
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)

# --- the freeze must be committed and unchanged; the frozen values govern -------------------
rel = os.path.relpath(os.path.abspath(freeze_path), os.getcwd())
SMOKE = os.environ.get("PREMISE_SMOKE") == "1"  # pre-freeze code test on SPARCS-derived stand-ins only
if not SMOKE and (subprocess.run(["git", "ls-files", "--error-unmatch", rel], capture_output=True).returncode
                  or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel]).returncode):
    sys.exit("freeze.json must be committed and unchanged before any HYPSO-1 inference")
freeze = json.load(open(freeze_path))
for k, v in freeze["config"].items():
    assert getattr(C, k) == v or (isinstance(v, list) and list(getattr(C, k)) == v), k
for k, h in freeze["model"]["member_sha256"].items():
    assert hashlib.sha256(open(f"{models}/member{k}.pt", "rb").read()).hexdigest() == h, f"member {k} changed"
members = [scoring.load_member(f"{models}/member{k}.pt", C.HIDDEN) for k in C.MEMBERS]
main, others = members[0], members[1:]
bounds = [tuple(b) for b in freeze["range_bounds"]]
thr = freeze["fault_thresholds"]
split = freeze["split"]


def read_list(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


captures = sorted(read_list(test_list))
reserved = set(read_list(reserved_list))
assert len(captures) == 38 and reserved <= set(captures) and len(reserved) == 10


def fmt(v):
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([fmt(v) for v in r.values()])


# --- source reference: the 16 SPARCS development subscenes ------------------------------------
src_rows = []
for n in sorted(split["dev"]):
    x = np.load(f"{sparcs_dir}/{n}_toa.npy"); lab = np.load(f"{sparcs_dir}/{n}_labels.npy")
    corners = tiles.grid(*lab.shape)
    s = gates.tile_scores(main, others, tiles.cut(x, corners), bounds)
    lab_frac, lab_down = scoring.label_tiles(tiles.cut(lab, corners), SPARCS_CLOUD, C.CLOUD_DECISION)
    down = scoring.implied_downlink(s["pred_cloud_fraction"], C.CLOUD_DECISION)
    for t in range(len(corners)):
        src_rows.append({"subscene": n, "tile": t, "label_cloud_frac": lab_frac[t],
                         "pred_cloud_frac": s["pred_cloud_fraction"][t], "correct": int(down[t] == lab_down[t]),
                         "conf": s["conf"][t], "ens": s["ens"][t], "violations": int(s["violations"][t])})
write_csv(f"{evid}/tiles-source-ref.csv", src_rows)

# --- shift-v0: 38 HYPSO-1 captures ---------------------------------------------------------------
tgt_rows, feats, clean = [], [], {}
cloud_pixels = cloud_saturated = 0
for cap in captures:
    x = np.load(f"{hypso_dir}/{cap}_toa.npy"); sat = np.load(f"{hypso_dir}/{cap}_sat.npy")
    name = cap.split("-", 1)[1]
    lab = np.fromfile(f"{download}/{cap}/{name}-slc-labels.bin", dtype=np.uint8).reshape(HYPSO_SHAPE)
    cloud_pixels += int((lab == HYPSO_CLOUD).sum()); cloud_saturated += int(((lab == HYPSO_CLOUD) & sat).sum())
    corners = tiles.grid(*HYPSO_SHAPE)
    xt = tiles.cut(x, corners)
    s = gates.tile_scores(main, others, xt, bounds, keep_logits=True)
    lt = tiles.cut(lab, corners).reshape(len(corners), -1)
    st = tiles.cut(sat, corners).reshape(len(corners), -1)
    lab_frac, lab_down = scoring.label_tiles(tiles.cut(lab, corners), HYPSO_CLOUD, C.CLOUD_DECISION)
    down = scoring.implied_downlink(s["pred_cloud_fraction"], C.CLOUD_DECISION)
    clean[cap] = {"x": xt, "logits": s["logits"], "down": down}
    for t in range(len(corners)):
        tgt_rows.append({"capture": cap, "tile": t, "reserved": int(cap in reserved),
                         "label_cloud_frac": lab_frac[t], "label_water_frac": float((lt[t] == 0).mean()),
                         "label_land_frac": float((lt[t] == 1).mean()), "sat_frac": float(st[t].mean()),
                         "pred_cloud_frac": s["pred_cloud_fraction"][t], "correct": int(down[t] == lab_down[t]),
                         "conf": s["conf"][t], "ens": s["ens"][t], "violations": int(s["violations"][t])})
        xp = xt[t].reshape(-1, 4)
        feats.append(np.concatenate([[s["conf"][t], s["ens"][t], math.log1p(s["violations"][t]),
                                      s["pred_cloud_fraction"][t]], s["mean_softmax"][t],
                                     xp.mean(0), xp.std(0), s["penultimate_mean"][t]]))
write_csv(f"{evid}/tiles-shift-v0.csv", tgt_rows)
np.savez(f"{runs}/probe-features.npz", X=np.array(feats), capture=np.array([r["capture"] for r in tgt_rows]))

# --- faults on shift-v0 tiles, main model ------------------------------------------------------
UNITS = [C.HIDDEN, C.HIDDEN, 3]


def detections(s, faulty_logits, clean_logits):
    conf_det = ~np.isfinite(s["conf"]) | (s["conf"] < thr["conf"])
    ens_det = ~np.isfinite(s["ens"]) | (s["ens"] < thr["ens"])
    range_det = s["violations"] > 0
    dup_det = np.array([not torch.equal(a, b) for a, b in zip(faulty_logits, clean_logits)])
    return {"conf": conf_det, "ens": ens_det, "range": range_det, "dup": dup_det}


def fault_row(kind, pas, cap, fid, layer, index, bit, harmful, det):
    r = {"class": kind, "pass": pas, "capture": cap, "fault": fid, "layer": layer, "index": index, "bit": bit,
         "n_tiles": int(len(harmful)), "n_harmful": int(harmful.sum())}
    for g in ("conf", "ens", "range", "dup"):
        r[f"harmful_det_{g}"] = int((harmful & det[g]).sum())
        r[f"benign_flag_{g}"] = int((~harmful & det[g]).sum())
    return r


def transient_pass(rng, pas):
    rows = []
    for cap in captures:
        c = clean[cap]
        for t in range(len(c["x"])):
            for j in range(C.TRANSIENT_PER_TILE):
                layer = int(rng.integers(3)); index = int(rng.integers(scoring.TILE_PIX * UNITS[layer]))
                bit = int(rng.integers(32))
                s = gates.tile_scores(main, others, c["x"][t:t + 1], bounds,
                                      hook=faults.transient_hook(layer, index, bit), keep_logits=True)
                down = scoring.implied_downlink(s["pred_cloud_fraction"], C.CLOUD_DECISION)
                harmful = down != c["down"][t:t + 1]
                det = detections(s, s["logits"], c["logits"][t:t + 1])
                rows.append(fault_row("transient", pas, cap, f"{t}.{j}", layer, index, bit, harmful, det))
    return rows


def persistent_pass(rng, pas):
    rows = []
    for cap in captures:
        c = clean[cap]
        for j in range(C.PERSISTENT_PER_CAPTURE):
            layer = int(rng.integers(3)); index = int(rng.integers(main.layers[layer].weight.numel()))
            bit = int(rng.integers(32))
            fm = faults.persistent_model(main, layer, index, bit)
            s = gates.tile_scores(fm, others, c["x"], bounds, keep_logits=True)
            again = scoring.forward_tiles(fm, c["x"])
            down = scoring.implied_downlink(s["pred_cloud_fraction"], C.CLOUD_DECISION)
            harmful = down != c["down"]
            det = detections(s, s["logits"], again)  # both runs carry the same weight fault
            rows.append(fault_row("persistent", pas, cap, str(j), layer, index, bit, harmful, det))
    return rows


fault_rows = []
for kind, fn, seed_k in (("transient", transient_pass, 0), ("persistent", persistent_pass, 1)):
    rng = np.random.default_rng([C.FAULT_SEED, seed_k])
    rows = fn(rng, 1)
    if sum(r["n_harmful"] for r in rows) < C.MIN_HARMFUL:
        rows += fn(rng, 2)
    fault_rows += rows
write_csv(f"{evid}/faults-shift-v0.csv", fault_rows)

# --- metrics ---------------------------------------------------------------------------------------
cap_arr = np.array([r["capture"] for r in tgt_rows])
correct = np.array([r["correct"] for r in tgt_rows], bool)
score = {"conf": np.array([r["conf"] for r in tgt_rows]), "ens": np.array([r["ens"] for r in tgt_rows])}
cov = C.COVERAGE
groups = [np.flatnonzero(cap_arr == c) for c in captures]
draws = metrics.cluster_draws(len(captures), C.BOOTSTRAP_DRAWS, C.BOOTSTRAP_SEED)


def boot(stat, groups_, draws_):
    return [stat(np.concatenate([groups_[i] for i in d])) for d in draws_]


res = {"n_captures": len(captures), "n_tiles": int(len(correct)), "target_accuracy": float(correct.mean()),
       "controls": {}, "model_checksums": freeze["model"]["member_sha256"]}
risk = {}
for g in ("conf", "ens"):
    pt = metrics.selective_risk(score[g], correct, cov)
    bs = boot(lambda i: metrics.selective_risk(score[g][i], correct[i], cov), groups, draws)
    au = metrics.auroc(score[g], correct)
    ba = boot(lambda i: metrics.auroc(score[g][i], correct[i]), groups, draws)
    res["controls"][g] = {"selective_risk": pt, "selective_risk_ci": metrics.percentile_interval(bs),
                          "auroc": au, "auroc_ci": metrics.percentile_interval(ba)}
    risk[g] = (pt, np.array(bs))
o_pt = metrics.oracle_risk(correct, cov)
o_bs = np.array(boot(lambda i: metrics.oracle_risk(correct[i], cov), groups, draws))
strongest = min(("conf", "ens"), key=lambda g: risk[g][0])
room_bs = risk[strongest][1] - o_bs
room_pt = risk[strongest][0] - o_pt
lo, hi = metrics.percentile_interval(room_bs)
outcome = "room" if lo >= 0.02 else ("no_room" if room_pt < 0.02 and hi < 0.05 else "inconclusive")
se = float(np.std(room_bs, ddof=1))
res.update({"oracle_risk": o_pt, "oracle_risk_ci": metrics.percentile_interval(o_bs), "strongest_control": strongest,
            "room": room_pt, "room_ci": [lo, hi], "room_bootstrap_se": se,
            "implied_between_capture_sd": se * math.sqrt(len(captures)), "outcome": outcome})

wrong = ~correct
acc70 = score["conf"] >= C.CONF_THRESHOLD
f1_bs = boot(lambda i: float((acc70[i] & wrong[i]).sum() / max(1, wrong[i].sum())), groups, draws)
f1 = float((acc70 & wrong).sum() / wrong.sum())
src_cap = np.array([r["subscene"] for r in src_rows])
src_correct = np.array([r["correct"] for r in src_rows], bool)
src_conf = np.array([r["conf"] for r in src_rows])
src_groups = [np.flatnonzero(src_cap == n) for n in sorted(split["dev"])]
src_draws = metrics.cluster_draws(len(src_groups), C.BOOTSTRAP_DRAWS, C.BOOTSTRAP_SEED + 1)
f3_bs = boot(lambda i: metrics.auroc(src_conf[i], src_correct[i]), src_groups, src_draws)
f1_ci, f2_ci, f3_ci = (metrics.percentile_interval(f1_bs), res["controls"]["conf"]["auroc_ci"],
                       metrics.percentile_interval(f3_bs))
res["gate_070"] = {"coverage": float(acc70.mean()), "risk": float(wrong[acc70].mean()) if acc70.any() else None,
                   "wrong_accepted_share": f1, "wrong_accepted_share_ci": f1_ci}
res["findings_paper"] = {
    "F1": {"value": f1, "ci": f1_ci, "holds": f1_ci[0] >= 0.50},
    "F2": {"value": res["controls"]["conf"]["auroc"], "ci": f2_ci, "holds": f2_ci[1] <= 0.75},
    "F3": {"value": metrics.auroc(src_conf, src_correct), "ci": f3_ci, "holds": f3_ci[0] >= 0.85},
}
res["findings_paper"]["all_hold"] = all(res["findings_paper"][k]["holds"] for k in ("F1", "F2", "F3"))

# secondary
res["source_reference"] = {"n_subscenes": len(src_groups), "n_tiles": int(len(src_correct)),
                           "accuracy": float(src_correct.mean()),
                           "auroc_conf": metrics.auroc(src_conf, src_correct),
                           "auroc_ens": metrics.auroc(np.array([r["ens"] for r in src_rows]), src_correct),
                           "range_false_reject": float(np.mean([r["violations"] > 0 for r in src_rows]))}
k_cap = math.ceil(cov * 140)
pc = {}
for g in ("conf", "ens"):
    wrong_acc = sum(int(wrong[gi][np.argsort(-score[g][gi], kind="stable")[:k_cap]].sum()) for gi in groups)
    pc[g] = wrong_acc / (k_cap * len(groups))
pc["oracle"] = sum(max(0, k_cap - int(correct[gi].sum())) for gi in groups) / (k_cap * len(groups))
pc["room"] = min(pc["conf"], pc["ens"]) - pc["oracle"]
res["per_capture_coverage"] = pc
res["saturation"] = {"cloud_pixels": cloud_pixels, "saturated_cloud_pixels": cloud_saturated,
                     "share": cloud_saturated / cloud_pixels}
res["range_check_clean_target_false_reject"] = float(np.mean([r["violations"] > 0 for r in tgt_rows]))
res["risk_coverage"] = {g: [[c_, metrics.selective_risk(score[g], correct, c_)] for c_ in np.round(np.arange(0.1, 1.01, 0.1), 2)]
                        for g in ("conf", "ens")}

fr = {}
overhead = {"conf": 1.0, "ens": 5.0, "range": 1.0, "dup": 2.0}
for kind in ("transient", "persistent"):
    rows = [r for r in fault_rows if r["class"] == kind]
    n_t = sum(r["n_tiles"] for r in rows); n_h = sum(r["n_harmful"] for r in rows)
    d = {"injections": len(rows), "faulted_tiles": n_t, "harmful": n_h, "harmful_share": n_h / n_t,
         "passes": max(r["pass"] for r in rows), "detection": {}, "benign_flag_rate": {}}
    for g in overhead:
        d["detection"][g] = sum(r[f"harmful_det_{g}"] for r in rows) / n_h if n_h else None
        d["benign_flag_rate"][g] = sum(r[f"benign_flag_{g}"] for r in rows) / (n_t - n_h) if n_t > n_h else None
    cheap = [d["detection"][g] for g in overhead if overhead[g] <= 1.05 and d["detection"][g] is not None]
    d["fault_room_cheap"] = 1 - max(cheap) if cheap else None
    if kind == "transient":
        d["accepted_harmful_per_10k_tiles"] = {str(rate): {g: 1e4 * rate * d["harmful_share"] * (1 - (d["detection"][g] or 0))
                                                           for g in overhead} for rate in C.RATES}
    fr[kind] = d
res["faults"] = fr
res["overhead_forward_passes"] = overhead

# room split by information and failure classes: inspectable captures only
insp = np.array([r["reserved"] == 0 for r in tgt_rows])
F = np.load(f"{runs}/probe-features.npz")["X"]
probe = np.full(len(correct), np.nan)
for c in sorted(set(cap_arr[insp])):
    tr = insp & (cap_arr != c); te = cap_arr == c
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(F[tr], correct[tr])
    probe[te] = clf.predict_proba(F[te])[:, 1]
r_ctrl = metrics.selective_risk(score[strongest][insp], correct[insp], cov)
r_probe = metrics.selective_risk(probe[insp], correct[insp], cov)
r_or = metrics.oracle_risk(correct[insp], cov)
res["probe_split_inspectable"] = {"n_captures": int(len(set(cap_arr[insp]))), "strongest_control_risk": r_ctrl,
                                  "probe_risk": r_probe, "oracle_risk": r_or,
                                  "share_of_room_probe_captures": (r_ctrl - r_probe) / (r_ctrl - r_or) if r_ctrl > r_or else None}
k_all = math.ceil(cov * len(correct))
accepted = np.zeros(len(correct), bool)
accepted[np.argsort(-score[strongest], kind="stable")[:k_all]] = True
fail = insp & accepted & wrong
lab_frac = np.array([r["label_cloud_frac"] for r in tgt_rows]); sat_frac = np.array([r["sat_frac"] for r in tgt_rows])
dom = np.argmax(np.stack([[r["label_water_frac"] for r in tgt_rows], [r["label_land_frac"] for r in tgt_rows], lab_frac]), axis=0)
bands = [(0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.01)]


def count(mask):
    return {"n": int(mask.sum()),
            "label_cloud_fraction": {f"{a}-{b}": int((mask & (lab_frac >= a) & (lab_frac < b)).sum()) for a, b in bands},
            "saturated_share": {"0": int((mask & (sat_frac == 0)).sum()), "(0,0.1]": int((mask & (sat_frac > 0) & (sat_frac <= 0.1)).sum()),
                                ">0.1": int((mask & (sat_frac > 0.1)).sum())},
            "dominant_label": {k: int((mask & (dom == i)).sum()) for i, k in enumerate(("water", "land", "cloud"))},
            "predicted_downlink": int((mask & (np.array([r["pred_cloud_frac"] for r in tgt_rows]) < C.CLOUD_DECISION)).sum())}


res["failure_classes_inspectable"] = {"accepted_wrong": count(fail), "all_inspectable_tiles": count(insp),
                                      "inspectable_wrong": count(insp & wrong)}
json.dump(res, open(f"{evid}/results.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("target_accuracy", "strongest_control", "room", "room_ci", "outcome")}, indent=1))
