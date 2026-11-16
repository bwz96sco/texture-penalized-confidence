"""Read-only full-development evaluation of fixed feature maps and gate recipes."""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perm_features import FEATURES, MATCHED, NESTED, PRIMARY, source_probability
from analysis import accepted, array, clean, inputs
from onboard import metrics

SETTINGS = (("hypso", "hypso", "A"), ("cloudsen_A", "cloudsen", "A"), ("cloudsen_B", "cloudsen", "B"))
REPLACEMENT = "spatial-replacement/v1"


def all_scores(rows, convention, freeze):
    main, spatial, truth, correct, scores, flags = inputs(rows, convention, freeze["legacy_freeze"])
    for name, fit in freeze["mappings"].items():
        scores[name] = scores["baseline"] - main * source_probability(rows, fit)
    supported = main & (array(rows, "q_real") > 0) & (array(rows, "dn_real") <= freeze["support_clear_95"])
    scores["support-filter/v1"] = scores["baseline"] - 1e3 * supported
    flags["support-filter/v1"] = supported
    known = [name for name in scores if name not in (PRIMARY, NESTED, REPLACEMENT)]
    return main, spatial, truth, correct, scores, flags, known


def swaps(base, before, after, current):
    common, drop, refill = base & after, base & ~after, ~base & after
    out = {"dropped_wrong": int((drop & ~before).sum()), "refill_wrong": int((refill & ~current).sum()),
           "dropped_correct": int((drop & before).sum()), "refill_correct": int((refill & current).sum()),
           "common_fixed_output": int((common & ~before & current).sum()),
           "common_broken_output": int((common & before & ~current).sum())}
    out["net_errors_removed"] = int((base & ~before).sum()) - int((after & ~current).sum())
    assert out["net_errors_removed"] == out["dropped_wrong"] - out["refill_wrong"] + out["common_fixed_output"] - out["common_broken_output"]
    return out


def summarize(rows, convention, freeze, timing=None):
    score_start = time.perf_counter()
    main, spatial, truth, correct, scores, flags, known = all_scores(rows, convention, freeze)
    if timing is not None:
        timing["score_generation_seconds"] = time.perf_counter() - score_start
    names = list(scores)
    masks = {name: accepted(score) for name, score in scores.items()}
    correctness = {name: spatial == truth if name == REPLACEMENT else correct for name in names}
    risks = {name: metrics.selective_risk(scores[name], correctness[name], .8) for name in names}
    best = min(known, key=lambda name: risks[name])
    base, k = masks["baseline"], int(masks["baseline"].sum())
    out = {"n_tiles": len(rows), "n_accepted": k, "n_units": len({r["unit"] for r in rows}),
           "n_clusters": len({r["cluster"] for r in rows}), "best_known": best,
           "best_known_risk": risks[best], "known_arms": known,
           "oracle_risk": metrics.oracle_risk(correct, .8), "arms": {}}
    for name in names:
        corr, acc = correctness[name], masks[name]
        pred = spatial if name == REPLACEMENT else main
        wrong = acc & ~corr
        out["arms"][name] = {"risk": risks[name], "n_accepted": int(acc.sum()),
            "n_accepted_wrong": int(wrong.sum()), "false_cloud": int((wrong & pred).sum()),
            "missed_cloud": int((wrong & ~pred).sum()),
            "correct_cloud_rejected": int((~acc & pred & corr).sum()),
            "baseline_correct_cloud_lost": int((base & correct & main & ~acc).sum()),
            "different_outputs": name == REPLACEMENT,
            "gain_to_baseline": 100 * (risks["baseline"] - risks[name]),
            "gain_to_matched_q": 100 * (risks[MATCHED] - risks[name]),
            "gain_to_best_known": 100 * (risks[best] - risks[name]),
            "transition": swaps(base, correct, acc, corr)}
        if name == REPLACEMENT:
            out["arms"][name]["accuracy"] = float(corr.mean())
            out["arms"][name]["own_oracle_risk"] = metrics.oracle_risk(corr, .8)
    clusters = np.asarray([r["cluster"] for r in rows])
    groups = [np.flatnonzero(clusters == name) for name in sorted(set(clusters))]
    draws = metrics.cluster_draws(len(groups), 2000, 7)
    boot = {name: [] for name in names}
    bootstrap_start = time.perf_counter()
    for draw in draws:
        idx = np.concatenate([groups[int(i)] for i in draw])
        for name in names:
            boot[name].append(metrics.selective_risk(scores[name][idx], correctness[name][idx], .8))
    boot = {name: np.asarray(value) for name, value in boot.items()}
    if timing is not None:
        timing["bootstrap_seconds"] = time.perf_counter() - bootstrap_start
    benchmark = np.min(np.stack([boot[name] for name in known]), axis=0)
    for name in names:
        arm = out["arms"][name]
        for reference, values in (("baseline", boot["baseline"]), ("matched_q", boot[MATCHED]), ("best_known", benchmark)):
            arm[f"gain_to_{reference}_ci"] = list(metrics.percentile_interval(100 * (values - boot[name])))
    out["nested_increment"] = {"gain_points": 100 * (risks["q-gradient/v1"] - risks[NESTED]),
        "gain_ci": list(metrics.percentile_interval(100 * (boot["q-gradient/v1"] - boot[NESTED])))}
    at70 = array(rows, "conf") >= .7
    out["confidence_070"] = {"coverage": float(at70.mean()), "risk": float((~correct)[at70].mean())}
    quartiles = np.quantile(array(rows, "ens")[main], [.25, .5, .75])
    bins = np.searchsorted(quartiles, array(rows, "ens"), side="right")
    out["associations"] = {}
    for group, mask in {"all_cloud_calls": main, **{f"base_quartile_{i}": main & (bins == i) for i in range(4)}}.items():
        out["associations"][group] = {"n": int(mask.sum()), "n_wrong": int((mask & ~correct).sum()),
            "error_auroc": {key: metrics.auroc(array(rows, key)[mask], (~correct)[mask])
                            for key in ("q_real", "a", "gradient", "q_multi3", "q_multiscale")}}
    old_gate_names = ("baseline", "lbp-continuous/v1", "lbp-filter/v1", "gradient-filter/v1",
                      "spatial-disagree/v1", "spatial-cloud-only/v1", REPLACEMENT)
    joint = np.logical_and.reduce([masks[name] & ~correctness[name] for name in old_gate_names])
    cloud_only = (array(rows, "dc_real") <= freeze["support_cloud_95"]) & (array(rows, "dn_real") > freeze["support_clear_95"])
    residual = joint & main & cloud_only
    out["old_cloud_compatible_residual"] = {"n": int(residual.sum()),
        "a_quantiles": np.quantile(array(rows, "a")[residual], [.05, .5, .95]).tolist() if residual.any() else None,
        "still_accepted_wrong": {name: int((residual & masks[name] & ~correctness[name]).sum()) for name in names}}
    if rows[0]["dataset"] == "hypso":
        keep = np.asarray([not r["unit"].startswith("111-bangladesh_") for r in rows])
        rr = {name: metrics.selective_risk(score[keep], correctness[name][keep], .8) for name, score in scores.items()}
        best_without = min(known, key=lambda name: rr[name])
        out["without_111"] = {"best_known": best_without, "best_known_risk": rr[best_without],
            "primary_risk": rr[PRIMARY], "primary_gain_to_best_known": 100 * (rr[best_without] - rr[PRIMARY])}
    return clean(out)


def analyze(rows, freeze, timings=None):
    out = {}
    for name, dataset, convention in SETTINGS:
        timing = {} if timings is not None else None
        out[name] = summarize([r for r in rows if r["dataset"] == dataset], convention, freeze, timing)
        if timings is not None:
            timings[name] = timing
    primary = {name: out[name]["arms"][PRIMARY] for name, _, _ in SETTINGS}
    qualifies = [name for name, arm in primary.items() if arm["gain_to_best_known"] >= 1 and arm["gain_to_best_known_ci"][0] > 0]
    if "hypso" in qualifies and all(primary[name]["gain_to_best_known"] >= 0 for name in ("cloudsen_A", "cloudsen_B")):
        outcome = "supports_HYPSO_recovery"
    elif qualifies:
        outcome = "sensor_specific_signal"
    elif any(arm["gain_to_best_known"] >= 1 for arm in primary.values()):
        outcome = "ambiguous_signal"
    elif any(arm["gain_to_best_known"] > 0 for arm in primary.values()):
        outcome = "weak_signal"
    else:
        outcome = "park_this_construction"
    out["decision"] = {"outcome": outcome, "primary": PRIMARY, "qualified_settings": qualifies,
        "best_known_selection": "minimum unchanged-base control risk, reselected within each cluster draw",
        "development_only": True, "new_method_pilot_authorized": False,
        "primary_gains_to_best_known": {name: arm["gain_to_best_known"] for name, arm in primary.items()}}
    return clean(out)
