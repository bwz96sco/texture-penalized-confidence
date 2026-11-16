"""Exact global-budget selection, net changes and conditional group resampling."""
import math

import numpy as np

from data import cut, group, key, values
from model import GUARD, select

METHODS = {'pairwise-v1': ('pairwise', GUARD), 'pointwise-ablation': ('pointwise', GUARD),
           'soft-budget-v2': ('soft-budget', GUARD), 'soft-budget-unguarded': ('soft-budget', 0.)}


def masks_for(offset, x, sat, raw, models):
    masks = {'baseline': cut(offset), 'ensemble': cut(raw[:, 0]), 'confidence-ranked': cut(raw[:, 1])}
    reports = {}
    for name, (kind, guard) in METHODS.items():
        if kind in models:
            masks[name], reports[name] = select(offset, x, sat, models[kind], guard)
    return masks, reports


def summarize(rows, offset, x, main, sat, error, models, n_draws=2000, progress=None):
    raw = values(rows, ('ens', 'conf'))
    masks, reports = masks_for(offset, x, sat, raw, models)
    base = masks['baseline']
    k = int(base.sum())
    counts = {name: int((mask & error).sum()) for name, mask in masks.items()}
    out = {'n_tiles': len(rows), 'n_groups': len({group(r) for r in rows}), 'n_accepted': k,
           'oracle_risk': max(0, k - int((~error).sum())) / k, 'arms': {}}
    for name, mask in masks.items():
        drop, refill = base & ~mask, ~base & mask
        wrong = mask & error
        transition = {'dropped_wrong': int((drop & error).sum()),
                      'refill_wrong': int((refill & error).sum()),
                      'dropped_correct': int((drop & ~error).sum()),
                      'refill_correct': int((refill & ~error).sum()),
                      'net_errors_removed': counts['baseline'] - counts[name]}
        assert transition['net_errors_removed'] == transition['dropped_wrong'] - transition['refill_wrong']
        out['arms'][name] = {'n_accepted': int(mask.sum()), 'n_accepted_wrong': counts[name],
            'risk': counts[name] / k, 'gain_to_baseline': 100 * (counts['baseline'] - counts[name]) / k,
            'gain_to_pointwise': 100 * (counts['pointwise-ablation'] - counts[name]) / k,
            'false_cloud': int((wrong & main).sum()), 'missed_cloud': int((wrong & ~main).sum()),
            'baseline_correct_cloud_lost': int((drop & ~error & main).sum()),
            'transition': transition, **reports.get(name, {})}
    groups = np.asarray([group(r) for r in rows])
    clusters = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    out['group_contributions'] = []
    for g, idx in zip(sorted(set(groups)), clusters):
        out['group_contributions'].append({'group': g, 'n_tiles': len(idx),
            'accepted': {name: int(mask[idx].sum()) for name, mask in masks.items()},
            'accepted_wrong': {name: int((mask[idx] & error[idx]).sum()) for name, mask in masks.items()}})
    if n_draws:
        rng = np.random.default_rng(7)
        draws = {name: [] for name in masks}
        for b in range(n_draws):
            picked = rng.integers(0, len(clusters), size=len(clusters))
            idx = np.sort(np.concatenate([clusters[int(i)] for i in picked]), kind='stable')
            resampled, _ = masks_for(offset[idx], x[idx], sat[idx], raw[idx], models)
            kk = math.ceil(.8 * len(idx))
            for name, mask in resampled.items():
                draws[name].append(float((mask & error[idx]).sum() / kk))
            if progress and (b + 1) % 500 == 0:
                progress({'paired_draws_done': b + 1})
        boot = {name: np.asarray(value) for name, value in draws.items()}
        for name in masks:
            out['arms'][name]['gain_to_baseline_ci'] = np.quantile(100 * (boot['baseline'] - boot[name]), [.025, .975]).tolist()
            out['arms'][name]['gain_to_pointwise_ci'] = np.quantile(100 * (boot['pointwise-ablation'] - boot[name]), [.025, .975]).tolist()
        out['bootstrap_risks'] = draws
    at70 = raw[:, 1] >= .7
    out['confidence_070_different_budget'] = {'coverage': float(at70.mean()),
                                            'risk': float(error[at70].mean()) if at70.any() else None}
    records = []
    for i, row in enumerate(rows):
        records.append({'key': key(row), 'group': group(row), 'correct': int(not error[i]),
                        **{name: int(mask[i]) for name, mask in masks.items()}})
    return out, records


def decisions(results):
    primary = results['cloudsen_A']['arms']['soft-budget-v2']
    required = results['cloudsen_B']['arms']['soft-budget-v2']
    benefit = primary['gain_to_baseline'] >= 1 and primary['gain_to_baseline_ci'][0] > 0
    increment = primary['gain_to_pointwise'] >= .5 and primary['gain_to_pointwise_ci'][0] > 0
    no_b_harm = min(required['gain_to_baseline'], required['gain_to_pointwise']) >= 0
    result = 'useful_development_mechanism' if benefit and increment and no_b_harm else (
        'ordinary_calibration_explains_benefit' if benefit and not increment else
        'inconclusive_or_not_supported')
    h = results['hypso']['arms']['soft-budget-v2']
    h_ok = (h['gain_to_baseline'] >= 1 and h['gain_to_baseline_ci'][0] > 0 and
            h['gain_to_pointwise'] >= .5 and h['gain_to_pointwise_ci'][0] > 0)
    return {'outcome': result, 'A_baseline_benefit': benefit, 'A_objective_increment': increment,
            'B_no_point_harm_to_both': no_b_harm, 'HYPSO_qualified': h_ok,
            'primary_version': 'soft-budget-v2', 'independent_confirmation': False,
            'novelty_established': False}
