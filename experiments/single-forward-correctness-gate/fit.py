"""Freeze exactly one two-feature head per setting, before held scoring."""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from protocol import FEATURES, OLD, PARENT, ROOT, data, execution, fit_heads, load


def main():
    started = time.perf_counter()
    evidence = Path(sys.argv[1]).resolve() / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    parent = load(PARENT / 'freeze.json')
    split = load(PARENT / 'splits.json')
    cols = parent['scale']['columns']
    indices = [cols.index(c) for c in FEATURES]
    scale = {'columns': list(FEATURES), 'mean': [parent['scale']['mean'][i] for i in indices],
             'sd': [parent['scale']['sd'][i] for i in indices], 'n_source_rows': 3600}
    source = data.values(data.read_csv(OLD / 'source-features.csv'), FEATURES)
    assert len(source) == 3600
    np.testing.assert_allclose(source.mean(0), scale['mean'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(source.std(0), scale['sd'], rtol=0, atol=1e-14)
    original = load(ROOT / 'tasks/texture-full-diagnostic/evidence/input-manifest.json')
    download = load(ROOT / 'tasks/cloudsen12-room/evidence/download.json')
    rows = data.read_csv(OLD / 'tiles.csv')
    held = {ds: take_ids['evaluation_keys'] for ds, take_ids in split['datasets'].items()}
    lookup = {data.key(r): r for r in rows}
    units = {ds: list(dict.fromkeys(lookup[k]['unit'] for k in ids)) for ds, ids in held.items()}
    assert set(units['hypso']) <= set(original['hypso_units'])
    assert set(units['cloudsen']) <= set(original['cloudsen_units'])
    assert (len(units['cloudsen']), len(units['hypso'])) == (455, 19)
    freeze = {'contract_revision': 1, 'features': list(FEATURES), 'scale': scale,
              'offset': 'zero', 'selection': 'stable global top ceil(0.8*N)',
              'loss': 'pointwise correctness cross-entropy', 'slope_l2': .05,
              'new_fits': 3, 'bootstrap_training_fits': 0,
              'historical_labels_per_sensor': 128, 'new_unique_labels': 0,
              'held_keys': held, 'units': units,
              'data_root': original['data_root'], 'cloudsen_file': original['cloudsen_file'],
              'models_directory': str(Path(original['data_root']) / 'runs/premise-test/models'),
              'padding_side': download['first_patch']['padding_side_row_col_start'],
              'runtime_order': ['single-1', 'ensemble-1', 'ensemble-2', 'single-2'],
              'torch_threads': 8, 'numpy_threads': 1, 'evaluation_draws': 2000,
              'evaluation_seed': 7, 'input_boundary': 'existing mapped four-band TOA inputs',
              'primary_quality': 'A positive paired lower endpoint vs both frozen CloudSEN gates; no B point harm',
              'primary_cost': 'single standalone S2-A time below ensemble on both passes'}
    heads = fit_heads(rows, split, scale)
    assert heads['cloudsen_A']['training_keys'] == heads['cloudsen_B']['training_keys']
    data.dump(evidence / 'freeze.json', freeze)
    data.dump(evidence / 'models.json', heads)
    data.dump(evidence / 'fit-execution.json', execution(started,
        new_head_fits=3, new_bootstrap_fits=0, new_main_fits=0, new_unique_labels=0,
        held_truth_converted=False, unused_commissioning_truth_converted=False, numerical_threads=1))
    print({name: {k: v for k, v in head.items() if k != 'training_keys'} for name, head in heads.items()}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
