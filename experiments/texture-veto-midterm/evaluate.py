"""One frozen development scoring: class-targeted texture veto on the zero-label rule.

Reads only the permutation premise's retained tiles, source features, freeze and
results, and the single-forward source scale. The one new fit is a source-only
class-blind texture mapping. Before any new arm is reported, the retained baseline
and multi-scale-template/v1 counts and that arm's gain interval must reproduce.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/texture-full-diagnostic'))
sys.path.insert(0, str(ROOT / 'experiments/texture-permutation-premise'))
from analysis import array, clean, inputs  # noqa: E402
from perm_features import SEED, feature_matrix, source_probability  # noqa: E402
from onboard import metrics  # noqa: E402

PREMISE = ROOT / 'tasks/texture-permutation-premise/evidence'
SETTINGS = (('hypso', 'hypso', 'A'), ('cloudsen_A', 'cloudsen', 'A'), ('cloudsen_B', 'cloudsen', 'B'))
RETAINED = 'multi-scale-template/v1'
FEATURE = 'q_multiscale'
VETO_THRESHOLD = 0.5  # frozen: source-calibrated error probability above which a cloud call is vetoed
A_WEIGHT = 0.5  # frozen zero-label rule, unchanged from tasks/zero-label-s2-test
GATES = {'confidence': 'softmax-response-cloudsen/v1', 'ensemble': 'ensemble-entropy-cloudsen/v1'}
COMPARISONS = (  # (arm, reference)
    ('veto-zl', 'zero-label'), ('veto-zl', 'confidence'), ('veto-zl', 'ensemble'), ('veto-zl', 'baseline'),
    ('veto-zl-blind', 'zero-label'), ('zero-label', 'confidence'), ('zero-label', 'ensemble'), ('zero-label', 'baseline'),
    (RETAINED, 'baseline'), ('texture-conf', 'confidence'), ('texture-conf-blind', 'confidence'),
    ('texture-conf-blind', 'texture-conf'))


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def cut(score):
    mask = np.zeros(len(score), bool)
    mask[np.argsort(-np.asarray(score), kind='stable')[:math.ceil(.8 * len(score))]] = True
    return mask


def fit_class_blind(source):
    assert len(source) == 3600
    x = feature_matrix(source, (FEATURE,))
    truth = np.asarray([1 - int(float(r['correct'])) for r in source])
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 0, scale, 1.0)
    fit = LogisticRegression(C=1, solver='lbfgs', tol=1e-8, max_iter=200, random_state=SEED).fit((x - mean) / scale, truth)
    assert fit.n_iter_[0] < 200
    entry = {'features': [FEATURE], 'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': fit.coef_[0].tolist(),
             'intercept': float(fit.intercept_[0]), 'iterations': int(fit.n_iter_[0]),
             'population': 'all 3,600 source-dev tiles', 'n_tiles': len(source), 'n_errors': int(truth.sum()),
             'n_parameters': 2}
    assert np.allclose(source_probability(source, entry), fit.predict_proba((x - mean) / scale)[:, 1], atol=1e-12)
    return entry


def scores_for(rows, convention, freeze, blind, scale):
    main, _, truth, correct, legacy, _ = inputs(rows, convention, freeze['legacy_freeze'])
    conf, cf, ens = array(rows, 'conf'), array(rows, 'pred_cloud_frac'), array(rows, 'ens')
    sat = (array(rows, 'sat_frac') > 0.1) & ~main
    p_cc = source_probability(rows, freeze['mappings'][RETAINED])
    p_all = source_probability(rows, blind)
    zero = A_WEIGHT * (conf - scale['mean'][0]) / scale['sd'][0] - (1 - A_WEIGHT) * (cf - scale['mean'][1]) / scale['sd'][1]
    veto = main & (p_cc > VETO_THRESHOLD)
    blind_veto = np.zeros(len(rows), bool)
    blind_veto[np.argsort(-p_all, kind='stable')[:int(veto.sum())]] = True
    scores = {'confidence': conf, 'ensemble': ens, 'baseline': legacy['baseline'],
              RETAINED: legacy['baseline'] - main * p_cc, 'zero-label': zero,
              'veto-zl': zero - 1e3 * veto, 'veto-zl-blind': zero - 1e3 * blind_veto,
              'texture-conf': (conf - 1e6 * sat) - main * p_cc, 'texture-conf-blind': (conf - 1e6 * sat) - p_all}
    assert np.array_equal(scores['ensemble'] - 1e6 * sat, scores['baseline'])
    flags = {'veto-zl': veto, 'veto-zl-blind': blind_veto}
    return main, truth, correct, scores, flags


def summarize(rows, convention, freeze, blind, scale, recorded):
    main, truth, correct, scores, flags = scores_for(rows, convention, freeze, blind, scale)
    error = ~correct
    masks = {name: cut(s) for name, s in scores.items()}
    arms = {}
    for name, m in masks.items():
        risk = metrics.selective_risk(scores[name], correct, .8)  # the retained results' risk definition
        assert abs(risk - float(error[m].mean())) < 1e-12
        arms[name] = {'n_accepted': int(m.sum()), 'n_accepted_wrong': int((m & error).sum()), 'risk': risk,
                      'false_cloud': int((m & main & ~truth).sum()), 'missed_cloud': int((m & ~main & truth).sum()),
                      'correct_cloud': int((m & main & truth).sum()), 'correct_clear': int((m & ~main & ~truth).sum())}
    for name in ('baseline', RETAINED):  # parity with the permutation premise before anything new is reported
        for key in ('n_accepted', 'n_accepted_wrong', 'risk'):
            assert arms[name][key] == recorded['arms'][name][key], (name, key, 'count parity')
    for name, f in flags.items():
        arms[name]['flags'] = {'n_flagged': int(f.sum()), 'flagged_cloud_calls': int((f & main).sum()),
                               'flagged_wrong': int((f & error).sum()), 'flagged_correct': int((f & ~error).sum()),
                               'flagged_correct_cloud': int((f & main & truth).sum())}
    clusters = np.asarray([r['cluster'] for r in rows])
    groups = [np.flatnonzero(clusters == c) for c in sorted(set(clusters))]
    boot = {name: [] for name in scores}
    for draw in metrics.cluster_draws(len(groups), 2000, 7):
        idx = np.concatenate([groups[int(i)] for i in draw])
        for name, s in scores.items():
            boot[name].append(metrics.selective_risk(s[idx], correct[idx], .8))
    boot = {name: np.asarray(v) for name, v in boot.items()}
    retained_ci = list(metrics.percentile_interval(100 * (boot['baseline'] - boot[RETAINED])))
    assert retained_ci == recorded['arms'][RETAINED]['gain_to_baseline_ci'], ('interval parity', retained_ci)
    comparisons = {}
    for arm, ref in COMPARISONS:
        entry = {'gain_points': 100 * (arms[ref]['risk'] - arms[arm]['risk']),
                 'gain_ci': list(metrics.percentile_interval(100 * (boot[ref] - boot[arm])))}
        if ref == 'zero-label':
            dropped, refilled = masks[ref] & ~masks[arm], masks[arm] & ~masks[ref]
            entry.update({'wrong_dropped': int((dropped & error).sum()), 'wrong_refilled': int((refilled & error).sum()),
                          'correct_dropped': int((dropped & ~error).sum()), 'correct_refilled': int((refilled & ~error).sum()),
                          'correct_cloud_dropped': int((dropped & ~error & main).sum())})
        comparisons[f'{arm} vs {ref}'] = entry
    at70 = array(rows, 'conf') >= .7
    retained_gain = comparisons[f'{RETAINED} vs baseline']['gain_points']
    return {'n_tiles': len(rows), 'n_clusters': len(groups), 'oracle_risk': metrics.oracle_risk(correct, .8),
            'arms': arms, 'comparisons': comparisons,
            'base_swap_kept_fraction': comparisons['texture-conf vs confidence']['gain_points'] / retained_gain,
            'conf_gate_070': {'coverage': float(at70.mean()), 'risk': float(error[at70].mean())}}


def decide(a):
    c = a['comparisons']
    gain, low = c['veto-zl vs zero-label']['gain_points'], c['veto-zl vs zero-label']['gain_ci'][0]
    if gain < 1.0 or low <= 0:
        return 'row 1, no useful increment: zero-label rule alone; texture reported as a known development check; no S2 test exposure'
    if not all(c[f'veto-zl vs {ref}']['gain_ci'][0] > 0 for ref in GATES):
        return 'row 2, control signal fails: zero-label rule alone; texture reported as a known development check'
    if c['veto-zl-blind vs zero-label']['gain_points'] > 0.5 * gain:
        return 'row 3, gain not carried by class targeting: report; the user decides the framing; no automatic S2 test'
    return "row 4, two-part method candidate: propose freezing veto-zl and one S2 test with complete timing, on the user's go"


def analyze(blind=None):
    freeze = json.loads((PREMISE / 'freeze.json').read_text())
    recorded = json.loads((PREMISE / 'results.json').read_text())
    scale = json.loads((ROOT / 'tasks/single-forward-correctness-gate/evidence/freeze.json').read_text())['scale']
    if blind is None:
        blind = fit_class_blind(read_csv(PREMISE / 'source-features.csv'))
    rows = read_csv(PREMISE / 'tiles.csv')
    assert len(rows) == 30135
    settings = {}
    for name, dataset, convention in SETTINGS:
        subset = [r for r in rows if r['dataset'] == dataset]
        settings[name] = summarize(subset, convention, freeze, blind, scale, recorded[name])
    return clean({'frozen': {'feature': FEATURE, 'cloud_call_mapping': RETAINED, 'veto_threshold': VETO_THRESHOLD,
                             'zero_label_weight': A_WEIGHT, 'scale': scale, 'selection': 'stable global top ceil(0.8*N)',
                             'bootstrap': 'cluster_draws(n, 2000, 7), unsorted concatenation, recomputed cut'},
                  'settings': settings, 'decision': decide(settings['cloudsen_A']),
                  'scope': 'development tiles only; already exposed; not confirmation'}), blind


def main(task):
    out = task / 'evidence'
    out.mkdir(exist_ok=True)
    assert not (out / 'results.json').exists(), 'One frozen scoring only.'
    started = time.perf_counter()
    blind = fit_class_blind(read_csv(PREMISE / 'source-features.csv'))
    (out / 'class-blind-fit.json').write_text(json.dumps(blind, indent=1))  # saved before target scoring
    result, _ = analyze(blind)
    (out / 'results.json').write_text(json.dumps(result, indent=1))
    (out / 'execution.json').write_text(json.dumps({
        'seconds': time.perf_counter() - started, 'scoring_batches': 1, 'source_only_fits': 1, 'target_fits': 0,
        'new_inference': 0, 'new_downloads': 0,
        'inputs': 'tasks/texture-permutation-premise/evidence (tiles, source features, freeze, results); '
                  'tasks/single-forward-correctness-gate/evidence/freeze.json (scale)'}, indent=1))
    for name, s in result['settings'].items():
        print(name, {arm: f"{v['n_accepted_wrong']} / {100 * v['risk']:.4f}%" for arm, v in s['arms'].items()})
        for label, c in s['comparisons'].items():
            print(f'  {label}: {c["gain_points"]:+.3f} [{c["gain_ci"][0]:+.3f}, {c["gain_ci"][1]:+.3f}]')
        print('  flags:', {k: s['arms'][k]['flags'] for k in ('veto-zl', 'veto-zl-blind')},
              'base-swap kept fraction:', round(s['base_swap_kept_fraction'], 3))
    print('decision:', result['decision'])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
