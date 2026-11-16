"""Read-only reconstruction of frozen training, replay, decisions and costs."""
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from protocol import FEATURES, OLD, PARENT, ROOT, data, decide, fit_heads, load
from evaluate import analyze


def main():
    evidence = Path(sys.argv[1]).resolve() / 'evidence'
    freeze, heads = load(evidence / 'freeze.json'), load(evidence / 'models.json')
    assert freeze['features'] == list(FEATURES) and freeze['offset'] == 'zero'
    splits = load(PARENT / 'splits.json')
    assert freeze['held_keys'] == {ds: s['evaluation_keys'] for ds, s in splits['datasets'].items()}
    rows = data.read_csv(OLD / 'tiles.csv')
    source = data.values(data.read_csv(OLD / 'source-features.csv'), FEATURES)
    np.testing.assert_allclose(source.mean(0), freeze['scale']['mean'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(source.std(0), freeze['scale']['sd'], rtol=0, atol=1e-14)
    assert fit_heads(rows, splits, freeze['scale']) == heads
    assert heads['cloudsen_A']['training_keys'] == heads['cloudsen_B']['training_keys']
    for h in heads.values():
        assert h['n_labels'] == len(set(h['training_keys'])) == 128 and len(h['theta']) == 2
    # Detect an accidental retained offset, guard or truth dependency.
    fixture = [{'conf': i, 'pred_cloud_frac': .5} for i in range(5)]
    scale = {'mean': [0, 0], 'sd': [1, 1]}
    h = {'theta': [1., 0.], 'intercept': 0.}
    mask = decide(fixture, scale, h)
    assert mask.tolist() == [False, True, True, True, True]
    spoiled = [{**r, 'ens': float('nan'), 'q_real': -1000, 'label_cloud_frac': 1000} for r in fixture]
    assert np.array_equal(mask, decide(spoiled, scale, h))
    h['theta'] = [0., 0.]
    assert decide(fixture, scale, h).tolist() == [True, True, True, True, False]
    reference = {data.key(r): r for r in rows}
    previous_features = {}
    for tag in freeze['runtime_order']:
        arm, repetition = tag.split('-')
        record = load(evidence / f'runtime-{tag}.json')
        features = data.read_csv(evidence / f'features-{tag}.csv')
        members = 1 if arm == 'single' else 5
        assert not record['smoke'] and record['arm'] == arm and str(record['repetition']) == repetition
        assert record['n_members'] == members and record['model_calls'] == [474] * members
        assert record['tile_equivalent_forwards'] == 24955 * members
        assert record['torch_threads'] == 8 and record['numerical_processes'] == 1
        assert not record['raw_labels_read'] and not record['truth_converted']
        assert not record['test_or_reserved_access'] and record['new_download_bytes'] == 0
        assert record['main_output_parity'] and record['operational_peak_rss_mib'] > 0
        assert record['operational_seconds'] >= record['setup_seconds'] + record['model_load_seconds'] + sum(
            d['total_seconds'] for d in record['datasets'].values()) - 1e-3
        assert record['standalone_cloudsen_A_seconds'] == record['setup_seconds'] + record['model_load_seconds'] + record['datasets']['cloudsen']['total_seconds']
        for dataset in ('cloudsen', 'hypso'):
            local = [r for r in features if r['dataset'] == dataset]
            assert [r['key'] for r in local] == freeze['held_keys'][dataset]
            units = [u for u in record['units'] if u['dataset'] == dataset]
            assert [u['unit'] for u in units] == freeze['units'][dataset]
            assert sum(u['n_tiles'] for u in units) == len(local)
            assert all(u['input_bytes_addressed'] > 0 for u in units)
            for field in ('io_seconds', 'preprocess_seconds', 'inference_seconds', 'collection_seconds'):
                assert abs(sum(u[field] for u in units) - record['datasets'][dataset][field]) < 1e-9
            if arm == 'single':
                assert 'ens' not in local[0] and 'sat_frac' not in local[0]
                setting = 'cloudsen_A' if dataset == 'cloudsen' else 'hypso'
                assert decide(local, freeze['scale'], heads[setting]).astype(int).tolist() == [int(r['accepted_primary']) for r in local]
                setting_b = 'cloudsen_B' if dataset == 'cloudsen' else 'hypso'
                assert decide(local, freeze['scale'], heads[setting_b]).astype(int).tolist() == [int(r['accepted_B']) for r in local]
            else:
                assert data.cut([float(r['ens']) for r in local]).astype(int).tolist() == [int(r['accepted_primary']) for r in local]
            for r in local:
                assert float(r['pred_cloud_frac']) == float(reference[r['key']]['pred_cloud_frac'])
                assert abs(float(r['conf']) - float(reference[r['key']]['conf'])) <= 2e-6
                if arm == 'ensemble':
                    assert abs(float(r['ens']) - float(reference[r['key']]['ens'])) <= 2e-6
                    assert float(r['sat_frac']) == float(reference[r['key']]['sat_frac'])
        if arm in previous_features:
            assert features == previous_features[arm], 'repeated replay features/decisions changed'
        previous_features[arm] = features
    result, decisions, groups, cost = analyze(evidence, lambda x: print(x, flush=True))
    assert result == load(evidence / 'results.json')
    assert cost == load(evidence / 'costs.json')
    for name, records in (('decisions.csv', decisions), ('groups.csv', groups)):
        assert data.read_csv(evidence / name) == [{k: str(v) for k, v in r.items()} for r in records]
    for setting, output in result['settings'].items():
        rr = [r for r in decisions if r['setting'] == setting]
        for arm, numbers in output['arms'].items():
            assert sum(r[arm] for r in rr) == output['n_accepted']
            assert sum(r[arm] * r['error'] for r in rr) == numbers['n_accepted_wrong']
            assert numbers['false_cloud'] + numbers['missed_cloud'] == numbers['n_accepted_wrong']
            assert sum(g[f'{arm}_wrong'] for g in groups if g['setting'] == setting) == numbers['n_accepted_wrong']
    fit, evaluation = load(evidence / 'fit-execution.json'), load(evidence / 'evaluation-execution.json')
    assert fit['new_head_fits'] == 3 and fit['new_bootstrap_fits'] == fit['new_main_fits'] == 0
    assert not fit['held_truth_converted'] and evaluation['frozen_control_parity']
    assert not fit['unused_commissioning_truth_converted'] and not evaluation['unused_commissioning_truth_converted']
    assert fit['new_unique_labels'] == evaluation['new_unique_labels'] == 0
    print('PASS: shared IDs, exact two-feature fit, four complete replays, 1/5 member calls, frozen controls, decisions, groups, paired intervals and cost accounting.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
