"""One frozen scoring of the zero-label confidence/cloud-fraction rule on the independent S2 test ROIs.

Reuses retained features, labels and the recorded evaluator protocol of
tasks/s2-independent-confirmation. No inference, fitting or download. Before any
new arm is reported, every recorded accept mask and bootstrap risk must reproduce.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tasks/s2-independent-confirmation/evidence'
GATES = {'confidence': 'softmax-response-cloudsen/v1', 'ensemble': 'ensemble-entropy-cloudsen/v1'}
A_WEIGHT = 0.5  # frozen: equal weight on source-standardized confidence and cloud fraction


def cut(score):
    accepted = np.zeros(len(score), bool)
    accepted[np.argsort(-np.asarray(score), kind='stable')[:math.ceil(.8 * len(score))]] = True
    return accepted


def zero_label_score(conf, cloud_fraction, scale):
    zc = (conf - scale['mean'][0]) / scale['sd'][0]
    zf = (cloud_fraction - scale['mean'][1]) / scale['sd'][1]
    return A_WEIGHT * zc - (1 - A_WEIGHT) * zf


def load_inputs():
    freeze = json.loads((SOURCE / 'freeze.json').read_text())
    prior = json.loads((ROOT / 'tasks/single-forward-correctness-gate/evidence/freeze.json').read_text())
    assert freeze['scale'] == prior['scale'], 'source scale must be the retained Landsat development scale'
    single = pd.read_csv(SOURCE / 'features-single-1.csv')
    ensemble = pd.read_csv(SOURCE / 'features-ensemble-1.csv')
    decisions = pd.read_csv(SOURCE / 'decisions.csv')
    recorded = json.loads((SOURCE / 'results.json').read_text())['settings']
    assert list(single.key) == list(ensemble.key) and len(single) == freeze['n_tiles'] == single.key.nunique()
    return freeze, single, ensemble, decisions, recorded


def analyze():
    freeze, single, ensemble, decisions, recorded = load_inputs()
    conf, cloud_fraction = single.conf.to_numpy(), single.pred_cloud_frac.to_numpy()
    main = cloud_fraction >= .5
    groups = single.cluster.to_numpy()
    clusters = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    zero = zero_label_score(conf, cloud_fraction, freeze['scale'])
    settings = {}
    for convention in ('A', 'B'):
        name = f'cloudsen_{convention}'
        rows = decisions[decisions.setting == name]
        assert list(rows.key) == list(single.key) and (rows.main.to_numpy() == main).all()
        error, truth = rows.error.to_numpy().astype(bool), rows.truth.to_numpy().astype(bool)
        head = freeze['heads'][name]
        z = (np.c_[conf, cloud_fraction] - freeze['scale']['mean']) / freeze['scale']['sd']
        score = {'single': z @ np.asarray(head['theta']) + head['intercept'], 'confidence': conf,
                 'ensemble': ensemble.ens.to_numpy(), 'zero_label': zero}
        accepted = {arm: cut(s) for arm, s in score.items()}
        for arm in ('single', 'confidence', 'ensemble'):
            assert (accepted[arm] == rows[arm].to_numpy().astype(bool)).all(), (name, arm, 'mask parity')
        arms = {arm: {'n_accepted': int(m.sum()), 'n_accepted_wrong': int((m & error).sum()),
                      'risk': float(error[m].mean()),
                      'false_cloud': int((m & main & ~truth).sum()), 'missed_cloud': int((m & ~main & truth).sum()),
                      'correct_cloud': int((m & main & truth).sum()), 'correct_clear': int((m & ~main & ~truth).sum())}
                for arm, m in accepted.items()}
        boot = {arm: [] for arm in score}
        rng = np.random.default_rng(freeze['bootstrap_seed'])
        for _ in range(freeze['bootstrap_draws']):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            ids = np.sort(np.concatenate([clusters[int(j)] for j in sampled]), kind='stable')
            for arm, s in score.items():
                boot[arm].append(float(error[ids][cut(s[ids])].mean()))
        for arm in ('single', 'confidence', 'ensemble'):
            assert boot[arm] == recorded[name]['bootstrap_risks'][arm], (name, arm, 'bootstrap parity')
        comparisons = {}
        for ref, label in (*GATES.items(), ('single', 'learned-head (reported arm)')):
            diff = 100 * (np.asarray(boot[ref]) - np.asarray(boot['zero_label']))  # positive: zero-label lower risk
            dropped, refilled = accepted[ref] & ~accepted['zero_label'], accepted['zero_label'] & ~accepted[ref]
            comparisons[label] = {'reference_arm': ref,
                                  'gain_points': 100 * (arms[ref]['risk'] - arms['zero_label']['risk']),
                                  'gain_ci': np.quantile(diff, [.025, .975]).tolist(),
                                  'wrong_dropped': int((dropped & error).sum()), 'wrong_refilled': int((refilled & error).sum()),
                                  'correct_dropped': int((dropped & ~error).sum()), 'correct_refilled': int((refilled & ~error).sum())}
        settings[name] = {'n_tiles': len(rows), 'n_groups': len(clusters), 'arms': arms,
                          'comparisons': comparisons, 'zero_label_bootstrap_risks': boot['zero_label']}
    a, b = settings['cloudsen_A']['comparisons'], settings['cloudsen_B']['comparisons']
    control_signal = all(a[g]['gain_ci'][0] > 0 for g in GATES.values()) and all(b[g]['gain_points'] >= 0 for g in GATES.values())
    head_clearly_better = any(s['comparisons']['learned-head (reported arm)']['gain_ci'][1] < 0 for s in settings.values())
    if not control_signal:
        decision = 'control signal fails: end the zero-label line at this scope; learned-head story stays'
    elif head_clearly_better:
        decision = 'control signal holds but learned head clearly better: learned head stays main; zero-label is a reported arm'
    else:
        decision = 'control signal holds and learned head not clearly better: zero-label rule eligible as proposed main method'
    return {'frozen_rule': {'a_weight_conf': A_WEIGHT, 'scale': freeze['scale'], 'selection': 'stable global top ceil(0.8*N)',
                            'direction': 'higher confidence and lower predicted cloud fraction accepted first'},
            'settings': settings, 'control_signal': control_signal, 'learned_head_clearly_better': head_clearly_better,
            'decision': decision, 'scope': 'independent S2 test ROIs already exposed to other arms; zero-label rule never scored here before'}


def main(task):
    out = task / 'evidence'
    assert not (out / 'results.json').exists(), 'One frozen scoring only.'
    started = time.perf_counter()
    result = analyze()
    (out / 'results.json').write_text(json.dumps(result, indent=1))
    (out / 'evaluation-execution.json').write_text(json.dumps({
        'seconds': time.perf_counter() - started, 'scoring_batches': 1, 'new_fits': 0, 'new_inference': 0,
        'new_downloads': 0, 'inputs': 'tasks/s2-independent-confirmation/evidence (features, decisions, freeze, results)'}, indent=1))
    for name, s in result['settings'].items():
        print(name, {arm: round(100 * v['risk'], 4) for arm, v in s['arms'].items()})
        for label, c in s['comparisons'].items():
            print(f'  vs {label}: gain {c["gain_points"]:+.4f} [{c["gain_ci"][0]:+.4f}, {c["gain_ci"][1]:+.4f}]')
    print({k: result[k] for k in ('control_signal', 'learned_head_clearly_better', 'decision')})


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
