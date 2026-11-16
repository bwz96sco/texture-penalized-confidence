"""Read-only reconstruction; detects policy/coverage/uncertainty/cost mismatches."""
import math
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from common import PRIOR, ROOT, data, decide, fixed, load, masks, verify_weights
from evaluate import analyze


def main(task):
    evidence = task / 'evidence'
    freeze, manifest = fixed(task)
    verify_weights(freeze)
    assert len(manifest) == freeze['n_patches']
    assert len({r['roi_id'] for r in manifest}) == freeze['n_rois']
    access = load(evidence / 'data-access.json')
    assert access['selected_patches'] == len(manifest)
    assert access['selected_rois'] == freeze['n_rois']
    assert len(manifest) + access['excluded_overlapping_roi_patches'] == access['official_test_509_patches'] == 975
    reference = data.read_csv(access['reference_metadata'])
    assert {r['roi_id'] for r in manifest}.isdisjoint({r['roi_id'] for r in reference})
    assert all(r['split'] == 'test' and str(r['proj_shape']) == '509' and r['label_type'] == 'high' for r in manifest)
    expected_keys = [f"cloudsen/{r['datapoint_id']}/{tile}" for r in manifest for tile in range(49)]
    all_features = {}
    for arm in ('single', 'ensemble'):
        repeats = []
        for repetition in (1, 2):
            rows = data.read_csv(evidence / f'features-{arm}-{repetition}.csv')
            repeats.append(rows)
            assert [r['key'] for r in rows] == expected_keys
            record = load(evidence / f'runtime-{arm}-{repetition}.json')
            n_members = 1 if arm == 'single' else 5
            assert record['n_members'] == n_members
            assert record['model_calls'] == [len(manifest)] * n_members
            assert record['tile_equivalent_forwards'] == len(expected_keys) * n_members
            assert not record['raw_labels_read'] and not record['truth_converted']
            assert record['imagery_bands_read'] == [1, 2, 3, 4]
            assert len(record['units']) == len(manifest)
            assert record['torch_threads'] == 8
            assert record['operational_seconds'] >= record['setup_seconds'] + record['model_load_seconds'] + record['pipeline']['total_seconds']
            for stage in ('io_seconds', 'preprocess_seconds', 'inference_seconds', 'collection_seconds'):
                assert np.isclose(sum(r[stage] for r in record['units']), record['pipeline'][stage], atol=1e-10)
            if arm == 'single':
                assert all('ens' not in r for r in rows)
                for convention, field in (('A', 'accepted_primary'), ('B', 'accepted_B')):
                    selected = decide(rows, freeze['scale'], freeze['heads'][f'cloudsen_{convention}'])
                    assert selected.tolist() == [bool(int(r[field])) for r in rows]
                    assert selected.sum() == math.ceil(.8 * len(rows))
            else:
                selected = data.cut(np.asarray([float(r['ens']) for r in rows]))
                assert selected.tolist() == [bool(int(r['accepted_primary'])) for r in rows]
                assert selected.tolist() == [bool(int(r['accepted_B'])) for r in rows]
        assert repeats[0] == repeats[1], ('repeated features/decisions differ', arm)
        all_features[arm] = repeats[0]
    for s, e in zip(all_features['single'], all_features['ensemble']):
        assert s['conf'] == e['conf'] and s['pred_cloud_frac'] == e['pred_cloud_frac']
    labels = data.read_csv(evidence / 'labels.csv')
    assert [r['key'] for r in labels] == expected_keys
    assert all(0 <= float(r['thick_frac']) <= 1 and 0 <= float(r['thin_frac']) <= 1 and float(r['thick_frac']) + float(r['thin_frac']) <= 1 for r in labels)
    truth_exec = load(evidence / 'truth-execution.json')
    assert truth_exec['bands_read'] == [14] and truth_exec['occurs_after_all_prediction_passes']
    assert all(load(evidence / f'runtime-{run}.json')['utc'] < truth_exec['utc'] for run in freeze['runtime_order'])
    assert load(evidence / 'download.json')['passes']
    transfer = load(evidence / 'transfer.json')
    assert transfer['complete'] and transfer['image_file_bytes'] == freeze['expected_file_bytes']
    assert transfer['accounted_payload_lower_bound_bytes'] == freeze['expected_file_bytes']
    result, decisions, groups, cost = analyze(task)
    assert result == load(evidence / 'results.json')
    assert cost == load(evidence / 'costs.json')
    assert [{k: str(v) for k, v in r.items()} for r in decisions] == data.read_csv(evidence / 'decisions.csv')
    assert [{k: str(v) for k, v in r.items()} for r in groups] == data.read_csv(evidence / 'groups.csv')
    for setting in result['settings'].values():
        for arm in setting['arms'].values():
            assert arm['n_accepted_wrong'] == arm['false_cloud'] + arm['missed_cloud']
            assert arm['n_accepted'] == arm['n_accepted_wrong'] + arm['correct_cloud'] + arm['correct_clear']
    print('Frozen heads, independent complete coverage, actual repeated inference, label-free decisions, global paired uncertainty and complete costs agree.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(Path(sys.argv[1]).resolve())
