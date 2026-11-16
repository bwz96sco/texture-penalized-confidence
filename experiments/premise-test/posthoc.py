"""Post-hoc analyses of the premise test's frozen records (not decision-bearing, not pre-fixed).

Usage: python -I posthoc.py EVIDENCE_DIR
Reads tiles-shift-v0.csv, faults-shift-v0.csv and freeze.json; writes posthoc.json. Every
per-capture quantity uses the 28 inspectable captures only; the reserved captures enter only
all-38 aggregates, as the pre-fixed metrics do.
"""
import csv, json, math, sys

import numpy as np

from onboard import metrics

evid = sys.argv[1]
fz = json.load(open(f"{evid}/freeze.json"))["config"]
cov, n_draws = fz["COVERAGE"], fz["BOOTSTRAP_DRAWS"]
tgt = list(csv.DictReader(open(f"{evid}/tiles-shift-v0.csv")))
cap = np.array([r["capture"] for r in tgt])
insp = np.array([r["reserved"] == "0" for r in tgt])
correct = np.array([r["correct"] == "1" for r in tgt])
wrong = ~correct
score = {g: np.array([float(r[g]) for r in tgt]) for g in ("conf", "ens")}
lab_cf = np.array([float(r["label_cloud_frac"]) for r in tgt])
pred_cf = np.array([float(r["pred_cloud_frac"]) for r in tgt])
sat = np.array([float(r["sat_frac"]) for r in tgt])
dom = np.argmax(np.stack([[float(r["label_water_frac"]) for r in tgt], [float(r["label_land_frac"]) for r in tgt], lab_cf]), axis=0)
out = {"note": "Post hoc, computed after the frozen run; not decision-bearing."}

# --- room under a per-capture budget (accept the best 80% of each capture's tiles) ----------------
captures = sorted(set(cap))
groups = [np.flatnonzero(cap == c) for c in captures]
k_cap = math.ceil(cov * 140)


def per_capture_counts(g, idx):
    return np.array([int(wrong[i][np.argsort(-score[g][i], kind="stable")[:k_cap]].sum()) for i in idx])


acc_w = {g: per_capture_counts(g, groups) for g in ("conf", "ens")}
orc_w = np.array([max(0, k_cap - int(correct[i].sum())) for i in groups])
strongest = min(("conf", "ens"), key=lambda g: acc_w[g].sum())
draws = metrics.cluster_draws(len(groups), n_draws, fz["BOOTSTRAP_SEED"])
room_bs = [(acc_w[strongest][d].sum() - orc_w[d].sum()) / (k_cap * len(d)) for d in draws]
room_pt = (acc_w[strongest].sum() - orc_w.sum()) / (k_cap * len(groups))
se = float(np.std(room_bs, ddof=1))
out["per_capture_budget_room_all38"] = {"control": strongest, "room": room_pt,
                                        "ci": metrics.percentile_interval(room_bs), "bootstrap_se": se,
                                        "implied_between_capture_sd": se * math.sqrt(len(groups)),
                                        "draws": "the frozen capture draws (BOOTSTRAP_SEED)"}

# the same split on the inspectable captures: global 80% cut vs per-capture 80% cut
ig = [i for c, i in zip(captures, groups) if insp[i[0]]]
ii = np.concatenate(ig)
glob_ctrl = metrics.selective_risk(score[strongest][ii], correct[ii], cov)
glob_orc = metrics.oracle_risk(correct[ii], cov)
pc_ctrl = per_capture_counts(strongest, ig).sum() / (k_cap * len(ig))
pc_orc = sum(max(0, k_cap - int(correct[i].sum())) for i in ig) / (k_cap * len(ig))
out["inspectable_room_global_vs_per_capture"] = {
    "n_captures": len(ig), "global": {"control": glob_ctrl, "oracle": glob_orc, "room": glob_ctrl - glob_orc},
    "per_capture": {"control": pc_ctrl, "oracle": pc_orc, "room": pc_ctrl - pc_orc}}

# --- inspectable captures: per-capture accuracy and the captures the model fails as a whole ------
per_cap = []
for c, i in zip(captures, groups):
    if not insp[i[0]]:
        continue
    per_cap.append({"capture": c, "tile_accuracy": float(correct[i].mean()),
                    "label_cloud_frac": float(lab_cf[i].mean()), "pred_cloud_frac": float(pred_cf[i].mean()),
                    "sat_frac": float(sat[i].mean()), "n_wrong": int(wrong[i].sum())})
per_cap.sort(key=lambda r: r["tile_accuracy"])
accs = np.array([r["tile_accuracy"] for r in per_cap])
out["inspectable_per_capture"] = {
    "tile_accuracy_quantiles": {q: float(np.quantile(accs, float(q))) for q in ("0", "0.25", "0.5", "0.75", "1")},
    "n_below_0.8": int((accs < 0.8).sum()), "n_below_0.5": int((accs < 0.5).sum()),
    "wrong_share_in_worst_5": sum(r["n_wrong"] for r in per_cap[:5]) / sum(r["n_wrong"] for r in per_cap),
    "captures": per_cap}

# --- failure classes on the inspectable captures, by error direction -----------------------------
accepted = np.zeros(len(tgt), bool)
accepted[np.argsort(-score[strongest], kind="stable")[:math.ceil(cov * len(tgt))]] = True
fail = insp & accepted & wrong
missed = lab_cf >= 0.5  # wrong with label cloudy: predicted clear, so the tile would be downlinked
sat_bands = {"0": sat == 0, "(0,0.1]": (sat > 0) & (sat <= 0.1), ">0.1": sat > 0.1}


def tab(mask):
    return {"n": int(mask.sum()),
            "saturated_share": {k: int((mask & m).sum()) for k, m in sat_bands.items()},
            "dominant_label": {k: int((mask & (dom == j)).sum()) for j, k in enumerate(("water", "land", "cloud"))},
            "label_cloud_below_0.1": int((mask & (lab_cf < 0.1)).sum()),
            "label_cloud_at_least_0.9": int((mask & (lab_cf >= 0.9)).sum())}


fail_caps = cap[fail]
by_cap = sorted(((int((fail_caps == c).sum()), c) for c in set(fail_caps)), reverse=True)
counts = [n for n, _ in by_cap]
out["accepted_wrong_inspectable"] = {
    "missed_cloud": tab(fail & missed), "false_cloud": tab(fail & ~missed),
    "acceptance_rate_of_wrong_tiles": {k: float((fail & m).sum() / (insp & wrong & m).sum()) for k, m in sat_bands.items()},
    "n_captures_with_any": len(counts), "share_in_top_5_captures": sum(counts[:5]) / sum(counts),
    "top_5_captures": [{"capture": c, "accepted_wrong": n, "missed_cloud": int((fail & missed & (cap == c)).sum())}
                       for n, c in by_cap[:5]]}
# sensitivity: room on the inspectable captures, 80% cut within them (as the probe split), without
# the capture whose visible channels are clipped
clipped = cap == "178-niger_2022-09-06T09-26-31Z"
whole = np.isin(cap, ["178-niger_2022-09-06T09-26-31Z", "111-bangladesh_2022-10-27T03-50-38Z",
                      "266-kotzbue_2022-06-25T22-10-13Z"])  # tile accuracy at most 0.05
for name, m in (("inspectable_28", insp), ("without_clipped_capture_27", insp & ~clipped),
                ("without_three_whole_capture_failures_25", insp & ~whole)):
    r_c, r_o = metrics.selective_risk(score[strongest][m], correct[m], cov), metrics.oracle_risk(correct[m], cov)
    out.setdefault("inspectable_room_sensitivity", {})[name] = {"control": r_c, "oracle": r_o, "room": r_c - r_o}

# --- persistent faults: concentration, per-faulted-capture accepted harm, injection-level intervals --
fr = [r for r in csv.DictReader(open(f"{evid}/faults-shift-v0.csv")) if r["class"] == "persistent"]
harm = np.array([int(r["n_harmful"]) for r in fr])
det = {g: np.array([int(r[f"harmful_det_{g}"]) for r in fr]) for g in ("conf", "ens", "range", "dup")}
n_tiles = np.array([int(r["n_tiles"]) for r in fr])
inj = np.random.default_rng([fz["BOOTSTRAP_SEED"], 2]).integers(0, len(fr), size=(n_draws, len(fr)))
harmful_inj = harm > 0
out["persistent_faults"] = {
    "injections": len(fr), "injections_with_harm": int(harmful_inj.sum()),
    "harmful_tiles_in_top_10_injections": int(np.sort(harm)[-10:].sum()), "harmful_tiles": int(harm.sum()),
    "harmful_injections_by_bit": {str(b): int(((np.array([int(r["bit"]) for r in fr]) == b) & harmful_inj).sum())
                                  for b in sorted({int(r["bit"]) for r, h in zip(fr, harmful_inj) if h})},
    "harmful_injections_by_layer": {str(l): int(((np.array([int(r["layer"]) for r in fr]) == l) & harmful_inj).sum())
                                    for l in range(3)},
    "per_faulted_capture_harmed_and_accepted_share": {g: float(((harm - det[g]) / n_tiles).mean()) for g in det},
    "detection_ci_resampling_injections": {g: metrics.percentile_interval([det[g][d].sum() / max(1, harm[d].sum()) for d in inj])
                                           for g in det},
    "share_of_harmful_injections_mostly_detected": {g: float(((det[g] / np.maximum(harm, 1)) >= 0.5)[harmful_inj].mean())
                                                    for g in det}}

# --- transient faults: none harmful, so a bound rather than a rate ------------------------------
n_tr = sum(1 for r in csv.DictReader(open(f"{evid}/faults-shift-v0.csv")) if r["class"] == "transient")
h_upper = 3 / n_tr  # rule of three, one-sided 95% with zero events
out["transient_faults"] = {"injections": n_tr, "harmful": 0, "harmful_share_upper_95": h_upper,
                           "accepted_harmful_per_10k_tiles_upper_95_no_detection":
                               {str(r): 1e4 * r * h_upper for r in fz["RATES"]}}

json.dump(out, open(f"{evid}/posthoc.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "inspectable_per_capture"}, indent=1))
