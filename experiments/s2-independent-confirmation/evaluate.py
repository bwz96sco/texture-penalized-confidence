"""One frozen independent comparison and paired whole-ROI uncertainty."""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from common import GATES, data, execution, fixed, load, scores, write_csv


def costs(evidence):
    runs = {run: load(evidence / f'runtime-{run}.json') for run in ('single-1', 'ensemble-1', 'ensemble-2', 'single-2')}
    arms = {}
    for arm in ('single', 'ensemble'):
        records = [runs[f'{arm}-{i}'] for i in (1, 2)]
        arms[arm] = {'median_seconds': float(np.median([r['operational_seconds'] for r in records])),
                     'maximum_peak_rss_mib': max(r['operational_peak_rss_mib'] for r in records)}
    access = load(evidence / 'data-access.json')
    transfer = load(evidence / 'transfer.json')
    return {'runtime_order': list(runs), 'runs': runs, 'by_arm': arms,
            'single_faster_on_both_passes': all(runs[f'single-{i}']['operational_seconds'] < runs[f'ensemble-{i}']['operational_seconds'] for i in (1, 2)),
            'median_speedup': arms['ensemble']['median_seconds'] / arms['single']['median_seconds'],
            'median_time_reduction_percent': 100 * (1 - arms['single']['median_seconds'] / arms['ensemble']['median_seconds']),
            'image_payload_bytes': access['image_download_bytes_expected'],
            'metadata_download_bytes': access['metadata_download_bytes'], 'transfer': transfer,
            'new_fits': 0, 'new_historical_labels': 0,
            'boundary': 'setup/model load and image-band read/decode through complete A decision; offline B/labels/statistics excluded',
            'platform_claim': 'local CPU, not flight hardware'}


def analyze(task, progress=None):
    evidence = task / 'evidence'
    freeze, manifest = fixed(task)
    single = data.read_csv(evidence / 'features-single-1.csv')
    ensemble = data.read_csv(evidence / 'features-ensemble-1.csv')
    labels = data.read_csv(evidence / 'labels.csv')
    keys = [r['key'] for r in single]
    assert keys == [r['key'] for r in ensemble] == [r['key'] for r in labels]
    assert len(keys) == freeze['n_tiles'] == len(set(keys))
    rows = [{**s, 'ens': e['ens']} for s, e in zip(single, ensemble)]
    main = np.asarray([float(r['pred_cloud_frac']) >= .5 for r in rows])
    groups = np.asarray([r['cluster'] for r in rows])
    names = sorted(set(groups))
    clusters = [np.flatnonzero(groups == group) for group in names]
    settings, decisions, group_rows = {}, [], []
    for convention in ('A', 'B'):
        truth = np.asarray([float(r['thick_frac']) + (float(r['thin_frac']) if convention == 'B' else 0) >= .5 for r in labels])
        error = main != truth
        score = scores(rows, freeze, convention)
        accepted = {k: data.cut(v) for k, v in score.items()}
        arms = {}
        for arm, mask in accepted.items():
            arms[arm] = {'n_accepted': int(mask.sum()), 'n_accepted_wrong': int((mask & error).sum()),
                         'risk': float(error[mask].mean()),
                         'false_cloud': int((mask & main & ~truth).sum()),
                         'missed_cloud': int((mask & ~main & truth).sum()),
                         'correct_cloud': int((mask & main & truth).sum()),
                         'correct_clear': int((mask & ~main & ~truth).sum())}
        bootstrap = {arm: [] for arm in score}
        rng = np.random.default_rng(freeze['bootstrap_seed'])
        for draw in range(freeze['bootstrap_draws']):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            ids = np.sort(np.concatenate([clusters[int(j)] for j in sampled]), kind='stable')
            for arm, values in score.items():
                mask = data.cut(values[ids])
                bootstrap[arm].append(float(error[ids][mask].mean()))
            if progress and (draw + 1) % 500 == 0:
                progress({'convention': convention, 'bootstrap_draws': draw + 1})
        comparisons = {}
        for control, gate_id in GATES.items():
            reference, method = accepted[control], accepted['single']
            dropped, refilled = reference & ~method, method & ~reference
            comparisons[gate_id] = {
                'gain_points': 100 * (arms[control]['risk'] - arms['single']['risk']),
                'gain_ci': (100 * np.quantile(np.asarray(bootstrap[control]) - bootstrap['single'], [.025, .975])).tolist(),
                'wrong_dropped': int((dropped & error).sum()), 'wrong_refilled': int((refilled & error).sum()),
                'correct_dropped': int((dropped & ~error).sum()), 'correct_refilled': int((refilled & ~error).sum()),
                'correct_cloud_lost': int((dropped & main & truth).sum())}
        settings[f'cloudsen_{convention}'] = {'n_patches': len(manifest), 'n_tiles': len(rows),
            'n_groups': len(clusters), 'arms': arms, 'comparisons': comparisons, 'bootstrap_risks': bootstrap}
        decisions.extend({'setting': f'cloudsen_{convention}', 'key': r['key'], 'group': r['cluster'],
                          'error': int(error[i]), 'truth': int(truth[i]), 'main': int(main[i]),
                          **{k: int(mask[i]) for k, mask in accepted.items()}}
                         for i, r in enumerate(rows))
        for group, ids in zip(names, clusters):
            group_rows.append({'setting': f'cloudsen_{convention}', 'group': group, 'n_tiles': len(ids),
                               **{f'{arm}_accepted': int(mask[ids].sum()) for arm, mask in accepted.items()},
                               **{f'{arm}_wrong': int((mask[ids] & error[ids]).sum()) for arm, mask in accepted.items()}})
        if progress:
            progress({'convention': convention, 'arms': arms, 'comparisons': comparisons})
    cost = costs(evidence)
    a_signal = all(v['gain_ci'][0] > 0 for v in settings['cloudsen_A']['comparisons'].values())
    b_signal = all(v['gain_points'] >= 0 for v in settings['cloudsen_B']['comparisons'].values())
    result = {'settings': settings, 'A_signal': a_signal, 'B_point_signal': b_signal,
              'quality_signal': a_signal and b_signal, 'cost_signal': cost['single_faster_on_both_passes'],
              'joint_confirmation_signal': a_signal and b_signal and cost['single_faster_on_both_passes'],
              'scope': 'independent test ROIs; fixed historical commissioning and heads; A/B share population',
              'decision': 'advance bounded S2 manuscript' if a_signal and b_signal and cost['single_faster_on_both_passes'] else 'end fixed-version main investment'}
    return result, decisions, group_rows, cost


def main(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    assert not (evidence / 'results.json').exists(), 'One frozen scored batch only.'
    result, decisions, groups, cost = analyze(task, lambda item: print(item, flush=True))
    data.dump(evidence / 'results.json', result)
    data.dump(evidence / 'costs.json', cost)
    write_csv(evidence / 'decisions.csv', decisions)
    write_csv(evidence / 'groups.csv', groups)
    data.dump(evidence / 'evaluation-execution.json', execution(started, n_tiles=result['settings']['cloudsen_A']['n_tiles'],
        new_fits=0, scoring_batches=1, fixed_head_bootstrap=True, test_used_for_training=False))
    print({k: v for k, v in result.items() if k != 'settings'}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(Path(sys.argv[1]).resolve())
