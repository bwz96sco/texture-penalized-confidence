"""One frozen scoring of the texture configuration on the 165 independent S2 test ROIs.

Reads the two image-only runtime passes, then the retained confirmation labels. Before
any texture arm is reported, the recorded gate masks, the gates' bootstrap risks and the
zero-label rule's bootstrap risks must reproduce exactly.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frozen import (CONFIRMATION, GATES, ROOT, VETO_THRESHOLD, ZERO_LABEL, cut, masks, matched, module,
                    p_cloud_call, scores, sha256)

INPUTS = module('supplementary_prepare', Path(__file__).parent / 'prepare.py').INPUTS

zl = module('supplementary_zero_label', ROOT / 'experiments/zero-label-s2-test/evaluate.py')
FREEZE_SHA256 = 'f9014a759b799260e4f67bddc5b94edc7329f48c277ae6b818ec7b821df27d59'
COMPARISONS = (  # (arm, reference); gain = reference risk - arm risk
    ('texture-conf', 'confidence'), ('texture-conf', 'ensemble'), ('texture-conf', 'zero-label'),
    ('texture-conf', 'matched-control'), ('texture-conf', 'veto-zl'),
    ('veto-zl', 'confidence'), ('veto-zl', 'ensemble'), ('veto-zl', 'zero-label'),
    ('zero-label', 'confidence'), ('zero-label', 'ensemble'), ('matched-control', 'confidence'))


def load(task):
    evidence = task / 'evidence'
    assert sha256(evidence / 'freeze.json') == FREEZE_SHA256
    freeze = json.loads((evidence / 'freeze.json').read_text())
    for name, digest in freeze['input_sha256'].items():
        assert sha256(INPUTS[name]) == digest, name
    runs = [json.loads((evidence / f'runtime-{i}.json').read_text()) for i in (1, 2)]
    assert all(r['freeze_sha256'] == FREEZE_SHA256 and not r['raw_labels_read'] for r in runs)
    with open(evidence / 'features.csv', newline='') as f:
        features = list(csv.DictReader(f))
    return freeze, runs, features


def costs(runs):
    confirmation = json.loads((CONFIRMATION / 'costs.json').read_text())['by_arm']
    median = lambda f: float(np.median([f(r) for r in runs]))
    steps = {k: median(lambda r, k=k: r['pipeline'][k]) for k in runs[0]['pipeline']}
    texture_cloud = steps['texture_seconds'] + steps['mapping_seconds']
    texture_all = median(lambda r: r['reference_all_tile_texture_seconds'])
    return {'passes': len(runs), 'median_operational_seconds': median(lambda r: r['operational_seconds']),
            'median_pipeline_steps_seconds': steps,
            'maximum_peak_rss_mib': max(r['operational_peak_rss_mib'] for r in runs),
            'texture_and_mapping_cloud_calls_seconds': texture_cloud,
            'texture_all_tiles_reference_seconds': texture_all,
            'cloud_calls_textured': runs[0]['n_cloud_calls_textured'], 'n_tiles': runs[0]['n_tiles'],
            'texture_ms_per_cloud_call': 1000 * steps['texture_seconds'] / runs[0]['n_cloud_calls_textured'],
            'texture_share_of_pipeline': texture_cloud / steps['total_seconds'],
            'retained_single_forward_median_seconds': confirmation['single']['median_seconds'],
            'retained_ensemble_median_seconds': confirmation['ensemble']['median_seconds'],
            'hardware': runs[0]['hardware'],
            'boundary': 'model load and band read/decode through the complete global texture-conf decision; '
                        'reference all-tile texture, parity checks, labels and statistics excluded',
            'platform_claim': 'local CPU, not flight hardware'}


def analyze(task):
    freeze, runs, features = load(task)
    _, single, ensemble, decisions, recorded = zl.load_inputs()
    zero_recorded = json.loads((ZERO_LABEL / 'results.json').read_text())['settings']
    assert list(single.key) == [r['key'] for r in features]
    conf, cloud_fraction, ens = single.conf.to_numpy(), single.pred_cloud_frac.to_numpy(), ensemble.ens.to_numpy()
    main = cloud_fraction >= .5
    q = np.asarray([float(r['q_multiscale']) if r['q_multiscale'] else np.nan for r in features])
    assert np.isfinite(q[main]).all() and np.isnan(q[~main]).all()
    p_cc = np.zeros(len(q))
    p_cc[main] = p_cloud_call(q[main], freeze['cloud_call_mapping'])
    sat_frac = np.asarray([float(r['sat_frac']) for r in features])
    _, base, veto, score = scores(conf, cloud_fraction, ens, sat_frac, p_cc, freeze['scale'])
    _, _, accepted = masks(conf, cloud_fraction, ens, sat_frac, p_cc, freeze['scale'])
    clear = ~main
    assert np.array_equal(accepted['matched-control'] & clear, accepted['texture-conf'] & clear)
    runtime_mask = np.asarray([r['accepted_texture_conf'] == '1' for r in features])
    exact_conf = all(r['parity_with_retained_features']['conf'] == 0 for r in runs)
    if exact_conf:
        assert np.array_equal(runtime_mask, accepted['texture-conf'])
    groups = single.cluster.to_numpy()
    clusters = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    settings = {}
    for convention in ('A', 'B'):
        name = f'cloudsen_{convention}'
        rows = decisions[decisions.setting == name]
        assert list(rows.key) == list(single.key) and (rows.main.to_numpy() == main).all()
        error, truth = rows.error.to_numpy().astype(bool), rows.truth.to_numpy().astype(bool)
        for arm in GATES:
            assert np.array_equal(accepted[arm], rows[arm].to_numpy().astype(bool)), (name, arm, 'mask parity')
        arms = {}
        for arm, m in accepted.items():
            arms[arm] = {'n_accepted': int(m.sum()), 'n_accepted_wrong': int((m & error).sum()),
                         'risk': float(error[m].mean()), 'n_accepted_cloud_calls': int((m & main).sum()),
                         'false_cloud': int((m & main & ~truth).sum()), 'missed_cloud': int((m & ~main & truth).sum()),
                         'correct_cloud': int((m & main & truth).sum()), 'correct_clear': int((m & ~main & ~truth).sum())}
        zr = zero_recorded[name]['arms']['zero_label']
        assert all(arms['zero-label'][k] == zr[k] for k in zr), (name, 'zero-label parity')
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
        for arm in GATES:
            assert boot[arm] == recorded[name]['bootstrap_risks'][arm], (name, arm, 'bootstrap parity')
        assert boot['zero-label'] == zero_recorded[name]['zero_label_bootstrap_risks'], (name, 'zero-label bootstrap parity')
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
        settings[name] = {'n_tiles': len(rows), 'n_groups': len(clusters), 'arms': arms, 'comparisons': comparisons,
                          'bootstrap_risks': {arm: boot[arm] for arm in ('texture-conf', 'veto-zl', 'matched-control')}}
    decision, mechanism = decide(settings)
    return {'frozen': {k: freeze[k] for k in ('arms', 'primary', 'selection', 'veto_budget_rule', 'texture_scope',
                                              'bootstrap', 'constants')},
            'freeze_sha256': FREEZE_SHA256, 'n_cloud_calls_textured': int(main.sum()),
            'n_saturation_flags': int(((sat_frac > freeze['constants']['saturation_share']) & ~main).sum()),
            'veto_threshold': VETO_THRESHOLD, 'runtime_mask_equals_scored': bool(np.array_equal(runtime_mask, accepted['texture-conf'])),
            'settings': settings, 'costs': costs(runs), 'decision': decision, 'mechanism_wording': mechanism,
            'scope': 'pre-frozen supplementary evaluation on a used test set: the 165 ROIs were exposed through the '
                     'learned head, the gates and the zero-label rule; texture arms never scored here before'}


def decide(settings):
    a, b = settings['cloudsen_A']['comparisons'], settings['cloudsen_B']['comparisons']
    control = (all(a[f'texture-conf vs {g}']['gain_ci'][0] > 0 for g in GATES)
               and all(b[f'texture-conf vs {g}']['gain_points'] >= 0 for g in GATES))
    z = a['texture-conf vs zero-label']
    if not control:
        decision = ('row 1, control signal fails: the zero-label rule stays the recommended configuration; '
                    'texture-conf reported with this result; texture is development evidence only')
    elif z['gain_points'] < 1.0 or z['gain_ci'][0] <= 0 or b['texture-conf vs zero-label']['gain_points'] < 0:
        decision = ('row 2, no clear increment over the zero-label rule: the zero-label rule stays the recommended '
                    'configuration; texture-conf reported as an arm')
    else:
        decision = ('row 3, increment holds: texture-conf is the recommended configuration, reported as a '
                    'pre-frozen supplementary evaluation on a used test set; the zero-label rule is the key comparison arm')
    if a['texture-conf vs matched-control']['gain_ci'][0] > 0:
        mechanism = 'texture improves which cloud calls are accepted, beyond accepting fewer cloud calls'
    else:
        mechanism = ("texture's within-class contribution is not shown; the gain is not separated from "
                     'accepting fewer cloud calls')
    return decision, mechanism


def main(task):
    out = task / 'evidence'
    assert not (out / 'results.json').exists(), 'One frozen scoring only.'
    started = time.perf_counter()
    result = analyze(task)
    (out / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    (out / 'execution.json').write_text(json.dumps({
        'seconds': time.perf_counter() - started, 'scoring_batches': 1, 'new_fits': 0,
        'labels_first_read_here': True, 'freeze_sha256': FREEZE_SHA256,
        'inputs': 'evidence/features.csv and runtime-1/2.json; tasks/s2-independent-confirmation/evidence '
                  '(features, decisions, freeze, results, costs); tasks/zero-label-s2-test/evidence/results.json'}, indent=1) + '\n')
    for name, s in result['settings'].items():
        print(name, {arm: f"{v['n_accepted_wrong']} / {100 * v['risk']:.4f}%" for arm, v in s['arms'].items()})
        for label, c in s['comparisons'].items():
            print(f'  {label}: {c["gain_points"]:+.3f} [{c["gain_ci"][0]:+.3f}, {c["gain_ci"][1]:+.3f}]')
    print('veto flags:', result['settings']['cloudsen_A']['arms']['veto-zl']['flags'])
    print('decision:', result['decision'])
    print('mechanism:', result['mechanism_wording'])
    print('costs:', json.dumps(result['costs'], indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
