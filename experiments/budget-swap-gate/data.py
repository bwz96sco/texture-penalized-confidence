"""Trusted, retained features and label-blind commissioning/evaluation roles."""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'tasks/texture-permutation-premise/evidence'
FEATURES = ('conf', 'pred_cloud_frac', 'q_real', 'dc_real', 'dn_real',
            'gradient', 'q_multi1', 'q_multi3', 'q_multi9')
SEED = 20261016
SETTINGS = (('cloudsen_A', 'cloudsen', 'A'), ('cloudsen_B', 'cloudsen', 'B'),
            ('hypso', 'hypso', 'A'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def key(row):
    return f"{row.get('dataset', 'source')}/{row['unit']}/{row['tile']}"


def group(row):
    if row['dataset'] == 'hypso':
        return row['unit'].split('_', 1)[0].split('-', 1)[1].lower()
    return row['cluster']


def values(rows, columns):
    result = np.asarray([[float(r[c]) for c in columns] for r in rows])
    assert np.isfinite(result).all()
    return result


def scale_source(rows):
    cols = ('ens',) + FEATURES
    x = values(rows, cols)
    sd = x.std(0)
    assert (sd > 0).all() and len(rows) == 3600
    return {'columns': list(cols), 'mean': x.mean(0).tolist(), 'sd': sd.tolist(),
            'n_source_rows': len(rows), 'label_access': 'none for scaling'}


def arrays(rows, scale):
    cols = tuple(scale['columns'])
    z = (values(rows, cols) - np.asarray(scale['mean'])) / np.asarray(scale['sd'])
    main = values(rows, ('pred_cloud_frac',))[:, 0] >= .5
    sat = (values(rows, ('sat_frac',))[:, 0] > .1) & ~main
    return z[:, 0] - 1e6 * sat, z[:, 1:], main, sat


def errors(rows, convention):
    main = values(rows, ('pred_cloud_frac',))[:, 0] >= .5
    truth = np.asarray([float(r['label_cloud_frac']) +
                        (float(r.get('thin_frac', 0)) if convention == 'B' else 0)
                        >= .5 for r in rows])
    return main != truth, truth


def cut(score):
    result = np.zeros(len(score), bool)
    result[np.argsort(-np.asarray(score), kind='stable')[:math.ceil(.8 * len(score))]] = True
    return result


def make_splits(rows):
    result = {'seed': SEED, 'datasets': {}}
    for dataset, n_cal, n_per in (('cloudsen', 16, 8), ('hypso', 8, 16)):
        pop = [r for r in rows if r['dataset'] == dataset]
        names = sorted({group(r) for r in pop})
        order = np.random.default_rng(SEED).permutation(names).tolist()
        cal_groups, eval_groups = order[:n_cal], order[n_cal:]
        selected = []
        for name in cal_groups:
            units = sorted({r['unit'] for r in pop if group(r) == name})
            queues = [[key(r) for r in pop if group(r) == name and r['unit'] == unit]
                      for unit in units]
            count = 0
            for tile_slot in range(max(map(len, queues))):
                for queue in queues:
                    if tile_slot < len(queue) and count < n_per:
                        selected.append(queue[tile_slot])
                        count += 1
                if count == n_per:
                    break
            assert count == n_per
        assert len(selected) == 128 and len(set(selected)) == 128
        result['datasets'][dataset] = {
            'calibration_groups': cal_groups, 'evaluation_groups': eval_groups,
            'calibration_keys': selected,
            'evaluation_keys': [key(r) for r in pop if group(r) in eval_groups],
            'n_calibration_labels': 128, 'n_population_rows': len(pop),
            'n_unused_calibration_group_rows': sum(group(r) in cal_groups for r in pop) - 128}
    assert len(result['datasets']['cloudsen']['evaluation_groups']) == 91
    assert len(result['datasets']['hypso']['evaluation_groups']) == 13
    return result


def roles(rows, split):
    lookup = {key(r): r for r in rows}
    return ([lookup[k] for k in split['calibration_keys']],
            [lookup[k] for k in split['evaluation_keys']])
