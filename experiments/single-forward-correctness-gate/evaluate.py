"""Fixed-head quality and complete retained local-cost summary."""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from protocol import DRAWS, OLD, ROOT, SETTINGS, data, decide, execution, load, take, write_csv

GATE_IDS = {'cloudsen': {'confidence': 'softmax-response-cloudsen/v1',
                        'ensemble': 'ensemble-entropy-cloudsen/v1'},
            'hypso': {'confidence': 'conf-gate-0.70/v1', 'ensemble': 'selective-ood/v1',
                      'saturation': 'saturation-rule/v1'}}


def transition(reference, candidate, error):
    drop, refill = reference & ~candidate, candidate & ~reference
    assert drop.sum() == refill.sum()
    return {'dropped_wrong': int((drop & error).sum()), 'refill_wrong': int((refill & error).sum()),
            'dropped_correct': int((drop & ~error).sum()), 'refill_correct': int((refill & ~error).sum()),
            'net_errors_removed': int((drop & error).sum() - (refill & error).sum())}


def score_arrays(features, scale, head, dataset):
    from protocol import matrix
    scores = {'single': matrix(features, scale) @ np.asarray(head['theta']) + head['intercept'],
              'confidence': np.asarray([float(r['conf']) for r in features]),
              'ensemble': np.asarray([float(r['ens']) for r in features])}
    if dataset == 'hypso':
        main = np.asarray([float(r['pred_cloud_frac']) >= .5 for r in features])
        sat = np.asarray([float(r['sat_frac']) > .1 for r in features]) & ~main
        scores['saturation'] = scores['ensemble'] - 1e6 * sat
    return scores


def costs(evidence):
    runs = {name: load(evidence / f'runtime-{name}.json')
            for name in ('single-1', 'ensemble-1', 'ensemble-2', 'single-2')}
    output = {'runtime_order': list(runs), 'runs': {}, 'new_download_bytes': 0,
              'boundary': 'mapped-image file read through complete global decisions; imports/artifact writing excluded',
              'primary_convention': 'A', 'platform_claim': 'local CPU, not flight hardware'}
    for name, record in runs.items():
        output['runs'][name] = {k: record[k] for k in ('operational_seconds',
            'operational_peak_rss_mib', 'standalone_cloudsen_A_seconds', 'model_load_seconds',
            'setup_seconds', 'n_members', 'model_calls', 'tile_equivalent_forwards', 'datasets')}
    by_arm = {}
    for arm in ('single', 'ensemble'):
        records = [runs[f'{arm}-{i}'] for i in (1, 2)]
        by_arm[arm] = {'median_operational_seconds': float(np.median([r['operational_seconds'] for r in records])),
                      'median_standalone_cloudsen_A_seconds': float(np.median([r['standalone_cloudsen_A_seconds'] for r in records])),
                      'maximum_operational_peak_rss_mib': max(r['operational_peak_rss_mib'] for r in records)}
    output['by_arm'] = by_arm
    output['single_faster_on_both_passes'] = all(runs[f'single-{i}']['standalone_cloudsen_A_seconds'] <
                                               runs[f'ensemble-{i}']['standalone_cloudsen_A_seconds'] for i in (1, 2))
    output['cloudsen_A_median_speedup'] = by_arm['ensemble']['median_standalone_cloudsen_A_seconds'] / by_arm['single']['median_standalone_cloudsen_A_seconds']
    output['full_median_speedup'] = by_arm['ensemble']['median_operational_seconds'] / by_arm['single']['median_operational_seconds']
    return output


def analyze(evidence, progress=None):
    freeze, heads = load(evidence / 'freeze.json'), load(evidence / 'models.json')
    prior = load(ROOT / 'tasks/pool-aware-gate/evidence/results.json')['settings']
    reference = load(ROOT / 'tasks/correctness-gate-spatial-ablation/evidence/results.json')['settings']
    all_rows = data.read_csv(OLD / 'tiles.csv')
    single = data.read_csv(evidence / 'features-single-1.csv')
    ensemble = data.read_csv(evidence / 'features-ensemble-1.csv')
    lookup_s, lookup_e = {r['key']: r for r in single}, {r['key']: r for r in ensemble}
    settings, records, group_rows = {}, [], []
    for name, dataset, convention in SETTINGS:
        keys = freeze['held_keys'][dataset]
        rows = take(all_rows, keys)
        features = [{**lookup_e[key], 'conf': lookup_s[key]['conf'],
                     'pred_cloud_frac': lookup_s[key]['pred_cloud_frac']} for key in keys]
        scores = score_arrays(features, freeze['scale'], heads[name], dataset)
        masks = {arm: data.cut(score) for arm, score in scores.items()}
        # Only scoring after decisions converts held truth.
        error, truth = data.errors(rows, convention)
        main = np.asarray([float(r['pred_cloud_frac']) >= .5 for r in features])
        groups = np.asarray([data.group(r) for r in rows])
        clusters = [np.flatnonzero(groups == group) for group in sorted(set(groups))]
        k = int(masks['single'].sum())
        arms = {}
        for arm, accepted in masks.items():
            assert accepted.sum() == k
            arms[arm] = {'n_accepted': k, 'n_accepted_wrong': int((accepted & error).sum()),
                         'risk': float(error[accepted].mean()),
                         'false_cloud': int((accepted & main & ~truth).sum()),
                         'missed_cloud': int((accepted & ~main & truth).sum())}
        for control in GATE_IDS[dataset]:
            old_name = 'baseline' if control == 'saturation' else control
            assert arms[control]['n_accepted_wrong'] == prior[name]['arms'][old_name]['n_accepted_wrong'], (name, control)
        bootstrap = {arm: [] for arm in masks}
        rng = np.random.default_rng(7)
        for draw in range(DRAWS):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            ids = np.sort(np.concatenate([clusters[int(j)] for j in sampled]), kind='stable')
            # A fixed pointwise head's scores stay fixed; complete global cuts
            # are recomputed on the repeated rows, rather than resampling masks.
            sampled_masks = {arm: data.cut(score[ids]) for arm, score in scores.items()}
            for arm, accepted in sampled_masks.items():
                bootstrap[arm].append(float(error[ids][accepted].mean()))
            if progress and (draw + 1) % 500 == 0:
                progress({'setting': name, 'draws': draw + 1})
        comparisons = {}
        for control, gate_id in GATE_IDS[dataset].items():
            old_name = 'baseline' if control == 'saturation' else control
            assert bootstrap[control] == prior[name]['bootstrap_risks'][old_name], (name, control, 'bootstrap parity')
            comparisons[gate_id] = {'gain_points': 100 * (arms[control]['risk'] - arms['single']['risk']),
                'gain_ci': (100 * np.quantile(np.asarray(bootstrap[control]) - bootstrap['single'], [.025, .975])).tolist(),
                'transition': transition(masks[control], masks['single'], error),
                'correct_cloud_lost': int((masks[control] & ~masks['single'] & main & truth).sum())}
        literal = np.asarray([float(r['conf']) >= .7 for r in features])
        settings[name] = {'n_tiles': len(rows), 'n_groups': len(clusters), 'n_accepted': k,
            'arms': arms, 'comparisons': comparisons, 'bootstrap_risks': bootstrap,
            'literal_confidence_070': {'n_accepted': int(literal.sum()),
                'coverage': float(literal.mean()), 'n_accepted_wrong': int((literal & error).sum()),
                'risk': float(error[literal].mean()) if literal.any() else None},
            'historical_references': {arm: reference[name]['arms'][arm] for arm in ('full9', 'reduced2')}}
        for group, ids in zip(sorted(set(groups)), clusters):
            group_rows.append({'setting': name, 'group': group, 'n_tiles': len(ids),
                **{f'{arm}_accepted': int(masks[arm][ids].sum()) if arm in masks else ''
                   for arm in ('single', 'confidence', 'ensemble', 'saturation')},
                **{f'{arm}_wrong': int((masks[arm][ids] & error[ids]).sum()) if arm in masks else ''
                   for arm in ('single', 'confidence', 'ensemble', 'saturation')}})
        records.extend({'setting': name, 'key': data.key(r), 'group': data.group(r),
                        'error': int(error[i]), 'truth': int(truth[i]), 'main': int(main[i]),
                        'single': int(masks['single'][i]), 'confidence': int(masks['confidence'][i]),
                        'ensemble': int(masks['ensemble'][i]),
                        'saturation': int(masks['saturation'][i]) if 'saturation' in masks else ''}
                       for i, r in enumerate(rows))
        if progress:
            progress({'setting': name, 'errors': arms['single']['n_accepted_wrong'], 'comparisons': comparisons})
    quality = all(v['gain_ci'][0] > 0 for v in settings['cloudsen_A']['comparisons'].values()) and all(
        v['gain_points'] >= 0 for v in settings['cloudsen_B']['comparisons'].values())
    cc = costs(evidence)
    return {'settings': settings, 'quality_signal': quality, 'cost_signal': cc['single_faster_on_both_passes'],
            'joint_development_signal': quality and cc['single_faster_on_both_passes'],
            'scope': 'fixed commissioned heads, exposed development, paired ROI/site bootstrap',
            'main_campaign': 'remains paused; no novelty or flight-budget conclusion'}, records, group_rows, cc


def main():
    started = time.perf_counter()
    evidence = Path(sys.argv[1]).resolve() / 'evidence'
    result, decisions, groups, cost = analyze(evidence, lambda x: print(x, flush=True))
    data.dump(evidence / 'results.json', result)
    data.dump(evidence / 'costs.json', cost)
    write_csv(evidence / 'decisions.csv', decisions)
    write_csv(evidence / 'groups.csv', groups)
    data.dump(evidence / 'evaluation-execution.json', execution(started,
        held_truth_converted=True, unused_commissioning_truth_converted=False,
        new_unique_labels=0, numerical_threads=1, frozen_control_parity=True))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
