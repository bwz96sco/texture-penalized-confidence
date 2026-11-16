"""One frozen scoring on the untouched sample, with the supplementary evaluation's arms, comparisons and decision rule."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT  # noqa: E402
from frozen import cut, masks, matched, module, p_cloud_call, scores, sha256  # noqa: E402

supplementary = module('confirmation_supplementary_evaluate', ROOT / 'experiments/texture-s2-supplementary/evaluate.py')
COMPARISONS, decide = supplementary.COMPARISONS, supplementary.decide  # identical comparisons and decision rule
FREEZE_SHA256 = '8bfde43607cfdadc6c38fe1317264df054e85477898412d3ee68a18d0aeaecf7'


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def analyze(task):
    evidence = task / 'evidence'
    assert FREEZE_SHA256 and sha256(evidence / 'freeze.json') == FREEZE_SHA256
    freeze = json.loads((evidence / 'freeze.json').read_text())
    features, labels = read(evidence / 'features.csv'), read(evidence / 'labels.csv')
    assert [r['key'] for r in features] == [r['key'] for r in labels] and len(features) == freeze['population']['n_tiles']
    value = lambda key: np.asarray([float(r[key]) for r in features])
    conf, cloud_fraction, ens, sat_frac = value('conf'), value('pred_cloud_frac'), value('ens'), value('sat_frac')
    main = cloud_fraction >= .5
    q = np.asarray([float(r['q_multiscale']) if r['q_multiscale'] else np.nan for r in features])
    assert np.isfinite(q[main]).all() and np.isnan(q[~main]).all()
    p_cc = np.zeros(len(q))
    p_cc[main] = p_cloud_call(q[main], freeze['cloud_call_mapping'])
    _, base, veto, score = scores(conf, cloud_fraction, ens, sat_frac, p_cc, freeze['scale'])
    _, _, accepted = masks(conf, cloud_fraction, ens, sat_frac, p_cc, freeze['scale'])
    assert np.array_equal(accepted['matched-control'] & ~main, accepted['texture-conf'] & ~main)
    groups = np.asarray([r['cluster'] for r in features])
    clusters = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    thick, thin = (np.asarray([float(r[k]) for r in labels]) for k in ('thick_frac', 'thin_frac'))
    settings = {}
    for convention in ('A', 'B'):
        truth = (thick + (thin if convention == 'B' else 0)) >= .5
        error = main != truth
        arms = {}
        for arm, m in accepted.items():
            arms[arm] = {'n_accepted': int(m.sum()), 'n_accepted_wrong': int((m & error).sum()),
                         'risk': float(error[m].mean()), 'n_accepted_cloud_calls': int((m & main).sum()),
                         'false_cloud': int((m & main & ~truth).sum()), 'missed_cloud': int((m & ~main & truth).sum()),
                         'correct_cloud': int((m & main & truth).sum()), 'correct_clear': int((m & ~main & ~truth).sum())}
        arms['veto-zl']['flags'] = {'n_flagged': int(veto.sum()), 'share_of_tiles': float(veto.mean()),
                                    'exceeds_20_percent': bool(veto.mean() > .2),
                                    'flagged_wrong': int((veto & error).sum()), 'flagged_correct': int((veto & ~error).sum())}
        boot = {arm: [] for arm in accepted}
        rng = np.random.default_rng(freeze['bootstrap']['seed'])
        for _ in range(freeze['bootstrap']['draws']):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            ids = np.sort(np.concatenate([clusters[int(j)] for j in sampled]), kind='stable')
            drawn = {arm: cut(s[ids]) for arm, s in score.items()}
            drawn['matched-control'] = matched(base[ids], main[ids], drawn['texture-conf'])
            for arm, m in drawn.items():
                boot[arm].append(float(error[ids][m].mean()))
        comparisons = {}
        for arm, ref in COMPARISONS:
            diff = 100 * (np.asarray(boot[ref]) - np.asarray(boot[arm]))
            entry = {'gain_points': 100 * (arms[ref]['risk'] - arms[arm]['risk']),
                     'gain_ci': np.quantile(diff, [.025, .975]).tolist()}
            if ref == 'zero-label':
                dropped, refilled = accepted[ref] & ~accepted[arm], accepted[arm] & ~accepted[ref]
                entry.update({'wrong_dropped': int((dropped & error).sum()), 'wrong_refilled': int((refilled & error).sum()),
                              'correct_dropped': int((dropped & ~error).sum()),
                              'correct_refilled': int((refilled & ~error).sum())})
            comparisons[f'{arm} vs {ref}'] = entry
        settings[f'cloudsen_{convention}'] = {'n_tiles': len(features), 'n_groups': len(clusters), 'arms': arms,
                                              'comparisons': comparisons}
    decision, mechanism = decide(settings)
    return {'freeze_sha256': FREEZE_SHA256, 'n_cloud_calls_textured': int(main.sum()),
            'n_saturation_flags': int(((sat_frac > freeze['constants']['saturation_share']) & ~main).sum()),
            'settings': settings, 'decision': decision, 'mechanism_wording': mechanism,
            'scope': 'untouched CloudSEN12+ train-509-high ROIs (330 sampled); first scoring of any arm on them'}


def main(task):
    out = task / 'evidence'
    assert not (out / 'results.json').exists(), 'One frozen scoring only.'
    started = time.perf_counter()
    result = analyze(task)
    (out / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    (out / 'execution.json').write_text(json.dumps({'seconds': time.perf_counter() - started, 'scoring_batches': 1,
                                                    'new_fits': 0, 'freeze_sha256': FREEZE_SHA256}, indent=1) + '\n')
    for name, s in result['settings'].items():
        print(name, {arm: f"{v['n_accepted_wrong']} / {100 * v['risk']:.4f}%" for arm, v in s['arms'].items()})
        for label, c in s['comparisons'].items():
            print(f'  {label}: {c["gain_points"]:+.3f} [{c["gain_ci"][0]:+.3f}, {c["gain_ci"][1]:+.3f}]')
    print('decision:', result['decision'])
    print('mechanism:', result['mechanism_wording'])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
