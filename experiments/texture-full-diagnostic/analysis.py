"""Read-only scoring of retained per-tile records under the fixed protocol."""
import math

import numpy as np

from onboard import metrics

TEXTURE = ("lbp-continuous/v1", "lbp-filter/v1", "gradient-filter/v1")
SPATIAL = ("spatial-disagree/v1", "spatial-cloud-only/v1")
REPLACEMENT = "spatial-replacement/v1"


def array(rows, key):
    return np.asarray([float(r[key]) for r in rows])


def accepted(score):
    mask = np.zeros(len(score), bool)
    mask[np.argsort(-score, kind="stable")[:math.ceil(0.8 * len(score))]] = True
    return mask


def inputs(rows, convention, freeze):
    main = array(rows, "pred_cloud_frac") >= 0.5
    spatial = array(rows, "spatial_cloud_frac") >= 0.5
    truth = array(rows, "label_cloud_frac") + (array(rows, "thin_frac") if convention == "B" else 0) >= 0.5
    correct = main == truth
    sat = (array(rows, "sat_frac") > 0.1) & ~main
    ens, conf = array(rows, "ens"), array(rows, "conf")
    baseline = ens - 1e6 * sat
    q = array(rows, "lbp_q")
    flags = {"lbp-filter/v1": main & (array(rows, "lbp_dn") < array(rows, "lbp_dc")),
             "gradient-filter/v1": main & (array(rows, "gradient") > freeze["gradient_threshold"]),
             "spatial-disagree/v1": main != spatial,
             "spatial-cloud-only/v1": main & ~spatial}
    scores = {"ensemble": ens, "baseline": baseline, "confidence": conf,
              "lbp-continuous/v1": baseline - main * np.maximum(q, 0)}
    scores.update({k: baseline - 1e3 * v for k, v in flags.items()})
    scores[REPLACEMENT] = array(rows, "spatial_conf") - 1e6 * ((array(rows, "sat_frac") > 0.1) & ~spatial)
    return main, spatial, truth, correct, scores, flags


def rate(mask, population):
    return float(mask[population].mean()) if population.any() else None


def summarize(rows, convention, freeze):
    main, spatial, truth, correct, scores, flags = inputs(rows, convention, freeze)
    clusters = np.asarray([r["cluster"] for r in rows])
    names = sorted(set(clusters))
    groups = [np.flatnonzero(clusters == c) for c in names]
    draws = metrics.cluster_draws(len(groups), 2000, 7)
    base_acc = accepted(scores["baseline"])
    base_risk = metrics.selective_risk(scores["baseline"], correct, 0.8)
    out = {"n_units": len(set(r["unit"] for r in rows)), "n_clusters": len(names),
           "n_tiles": len(rows), "main_accuracy": float(correct.mean()),
           "oracle_risk": metrics.oracle_risk(correct, 0.8), "arms": {}}
    for name, score in scores.items():
        corr = (spatial == truth) if name == REPLACEMENT else correct
        pred = spatial if name == REPLACEMENT else main
        acc = accepted(score)
        wrong = acc & ~corr
        r = metrics.selective_risk(score, corr, 0.8)
        entry = {"risk": r, "gain_points": 100 * (base_risk - r),
                 "n_accepted": int(acc.sum()), "n_accepted_wrong": int(wrong.sum()),
                 "accepted_wrong_false_cloud": int((wrong & pred).sum()),
                 "accepted_wrong_missed_cloud": int((wrong & ~pred).sum()),
                 "correct_cloud_rejected": int((~acc & pred & corr).sum()),
                 "base_correct_cloud_lost": int((base_acc & correct & main & ~acc).sum()),
                 "base_correct_lost": int((base_acc & correct & ~acc).sum()),
                 "base_wrong_removed": int((base_acc & ~correct & ~acc).sum()),
                 "base_wrong_cloud_removed": int((base_acc & ~correct & main & ~acc).sum()),
                 "different_outputs": name == REPLACEMENT}
        if name in flags:
            flag = flags[name]
            entry["flag_rates"] = {"all": float(flag.mean()),
                                   "baseline_accepted_wrong": rate(flag, base_acc & ~correct),
                                   "baseline_accepted_correct": rate(flag, base_acc & correct),
                                   "correct_cloud": rate(flag, correct & main)}
        if name == REPLACEMENT:
            entry["accuracy"] = float(corr.mean())
            entry["own_oracle_risk"] = metrics.oracle_risk(corr, 0.8)
        out["arms"][name] = entry

    # Each resample repeats whole ROIs/captures and reapplies the global cut.
    boot = {name: [] for name in scores}
    for draw in draws:
        idx = np.concatenate([groups[int(i)] for i in draw])
        rb = metrics.selective_risk(scores["baseline"][idx], correct[idx], 0.8)
        for name, score in scores.items():
            if name == "baseline":
                boot[name].append(0.0)
            else:
                corr = ((spatial == truth) if name == REPLACEMENT else correct)[idx]
                boot[name].append(100 * (rb - metrics.selective_risk(score[idx], corr, 0.8)))
    for name in scores:
        out["arms"][name]["gain_ci"] = list(metrics.percentile_interval(boot[name]))

    at70 = array(rows, "conf") >= 0.7
    out["confidence_070"] = {"coverage": float(at70.mean()),
                             "risk": float((~correct)[at70].mean()) if at70.any() else None}
    cond = {}
    for feature in ("lbp_q", "gradient"):
        value = array(rows, feature)
        bins = np.quantile(array(rows, "ens")[main], [0.25, 0.5, 0.75]) if main.any() else []
        strata = np.searchsorted(bins, array(rows, "ens"), side="right")
        cond[feature] = {"cloud_calls": int(main.sum()),
                         "error_auroc": metrics.auroc(value[main], (~correct)[main]),
                         "base_score_quartiles": []}
        for k in range(4):
            mask = main & (strata == k)
            cond[feature]["base_score_quartiles"].append({"bin": k, "n": int(mask.sum()),
                 "wrong": int((mask & ~correct).sum()),
                 "error_auroc": metrics.auroc(value[mask], (~correct)[mask])})
    out["conditional_texture"] = cond
    out["base_score_quartile_boundaries"] = np.asarray(bins).tolist()
    if rows[0]["dataset"] == "hypso":
        keep = np.asarray([not r["unit"].startswith("111-bangladesh_") for r in rows])
        rb = metrics.selective_risk(scores["baseline"][keep], correct[keep], 0.8)
        out["without_111"] = {name: 100 * (rb - metrics.selective_risk(score[keep],
             ((spatial == truth) if name == REPLACEMENT else correct)[keep], 0.8))
             for name, score in scores.items()}
    return out


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def analyze(rows, freeze):
    hypso = [r for r in rows if r["dataset"] == "hypso"]
    cloudsen = [r for r in rows if r["dataset"] == "cloudsen"]
    assert len(hypso) == 3920 and len(cloudsen) == 26215
    assert len(set((r["dataset"], r["unit"], int(r["tile"])) for r in rows)) == len(rows)
    result = {"hypso": summarize(hypso, "A", freeze),
              "cloudsen_A": summarize(cloudsen, "A", freeze),
              "cloudsen_B": summarize(cloudsen, "B", freeze)}
    assert abs(result["hypso"]["arms"]["baseline"]["risk"] - 265 / 3136) < 1e-12
    assert result["cloudsen_A"]["arms"]["baseline"]["n_accepted_wrong"] == 4932
    assert result["cloudsen_B"]["arms"]["baseline"]["n_accepted_wrong"] == 4028
    a, b = result["cloudsen_A"]["arms"], result["cloudsen_B"]["arms"]
    useful = [k for k in TEXTURE if a[k]["gain_points"] >= 2 and b[k]["gain_points"] >= 0]
    best = max(TEXTURE, key=lambda k: a[k]["gain_points"])
    best_spatial = max(SPATIAL, key=lambda k: a[k]["gain_points"])
    if useful:
        decision = "useful_known_texture"
    elif any(max(a[k]["gain_points"], b[k]["gain_points"]) >= 2 for k in TEXTURE):
        decision = "convention_specific"
    elif max(max(a[k]["gain_points"], b[k]["gain_points"]) for k in TEXTURE) > 0:
        decision = "weak_separation"
    else:
        decision = "park_these_texture_constructions"
    result["decision"] = {"outcome": decision, "useful_texture_arms": useful,
                           "primary": "lbp-continuous/v1", "best_A_texture": best,
                           "best_A_spatial_gate": best_spatial,
                           "spatial_control_adequate": freeze["spatial_control_adequate"],
                           "spatial_within_one_observed_point": bool(
                               freeze["spatial_control_adequate"] and a[best_spatial]["risk"] <= a[best]["risk"] + 0.01),
                           "useful_S2_arms_with_nonnegative_HYPSO_gain": [k for k in useful
                               if result["hypso"]["arms"][k]["gain_points"] >= 0],
                           "selection": "best of three known texture forms; development only"}
    return clean(result)
