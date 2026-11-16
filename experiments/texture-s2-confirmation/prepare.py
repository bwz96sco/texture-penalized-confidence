"""Write the confirmation freeze and prove the population-agnostic path on already-used patches.

The smoke reads two development patches and two of the 165 exposed test patches and must
reproduce their retained features. No image of the untouched population is read.
"""
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT, members, mlstac, patch  # noqa: E402
from frozen import sha256  # noqa: E402  (texture-s2-supplementary)

SUPPLEMENTARY = ROOT / 'tasks/texture-s2-supplementary/evidence'
POPULATION = ROOT / 'tasks/untouched-s2-population/evidence'
PREMISE = ROOT / 'tasks/texture-permutation-premise/evidence'
CONFIRMATION = ROOT / 'tasks/s2-independent-confirmation/evidence'


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def compare(fresh, expected, label):
    deltas = {}
    for key in ('conf', 'pred_cloud_frac', 'ens', 'q_multiscale'):
        got = np.asarray([f[key] for f in fresh])
        want = np.asarray([float(e[key]) if e.get(key) not in ('', None) else np.nan for e in expected])
        if key == 'q_multiscale':
            main = np.asarray([f['pred_cloud_frac'] >= .5 for f in fresh])
            assert np.isnan(got[~main]).all()
            got, want = got[main], want[main]
        deltas[key] = float(np.abs(got - want).max()) if len(got) else 0.0
    assert deltas['conf'] <= 2e-6 and deltas['ens'] <= 2e-6 and deltas['pred_cloud_frac'] == 0 \
        and deltas['q_multiscale'] == 0, (label, deltas)
    return deltas


def smoke(models, freeze):
    report = []
    manifest = json.loads((PREMISE / 'input-manifest.json').read_text())
    dev = [r for r in read(PREMISE / 'tiles.csv') if r['dataset'] == 'cloudsen']
    _, length, table = mlstac.header(manifest['cloudsen_file'])
    for name in manifest['cloudsen_units'][:2]:
        fresh = patch(models, manifest['cloudsen_file'], length, table[name], (0, 0),
                      freeze['prototypes'], freeze['cloud_call_mapping'])
        expected = sorted([r for r in dev if r['unit'] == name], key=lambda r: int(r['tile']))
        report.append({'population': 'development', 'unit': name, 'max_abs_delta': compare(fresh, expected, name)})
    test_file = json.loads((CONFIRMATION / 'freeze.json').read_text())['cloudsen_file']
    _, length, table = mlstac.header(test_file)
    single, ensemble = read(CONFIRMATION / 'features-single-1.csv'), read(CONFIRMATION / 'features-ensemble-1.csv')
    texture = {r['key']: r for r in read(SUPPLEMENTARY / 'features.csv')}
    for name in [r['datapoint_id'] for r in read(CONFIRMATION / 'manifest.csv')][:2]:
        fresh = patch(models, test_file, length, table[name], (0, 0), freeze['prototypes'], freeze['cloud_call_mapping'])
        keys = [f'cloudsen/{name}/{t}' for t in range(49)]
        s = {r['key']: r for r in single if r['unit'] == name}
        e = {r['key']: r for r in ensemble if r['unit'] == name}
        expected = [{**s[k], 'ens': e[k]['ens'], 'q_multiscale': texture[k]['q_multiscale']} for k in keys]
        report.append({'population': 'exposed test', 'unit': name, 'max_abs_delta': compare(fresh, expected, name)})
    return report


def main(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    evidence.mkdir(exist_ok=True)
    assert not (evidence / 'freeze.json').exists(), 'Do not replace the freeze.'
    base = json.loads((SUPPLEMENTARY / 'freeze.json').read_text())
    selection = json.loads((POPULATION / 'selection.json').read_text())
    transfer = json.loads((POPULATION / 'download.json').read_text())
    models = members(base['population']['models_directory'])
    freeze = {
        'task': 'tasks/texture-s2-confirmation', 'written_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
        'inherits': {'freeze': 'tasks/texture-s2-supplementary/evidence/freeze.json',
                     'sha256': sha256(SUPPLEMENTARY / 'freeze.json')},
        **{k: base[k] for k in ('scale', 'radii', 'prototypes', 'cloud_call_mapping', 'constants', 'arms', 'primary',
                                'selection', 'veto_budget_rule', 'texture_scope', 'bootstrap')},
        'population': {'manifest': 'tasks/untouched-s2-population/evidence/manifest.csv',
                       'manifest_sha256': sha256(POPULATION / 'manifest.csv'), 'file': transfer['local_path'],
                       'revision': transfer['revision'], 'n_patches': selection['counts']['manifest_patches'],
                       'n_rois': selection['counts']['sampled_rois'], 'n_tiles': selection['counts']['manifest_tiles'],
                       'padding_side': [0, 0], 'models_directory': base['population']['models_directory']},
        'truth': {'A': 'thick fraction >= 0.5', 'B': 'thick + thin fraction >= 0.5', 'band': 14,
                  'read': 'only after the prediction pass has written features.csv'}}
    with threadpool_limits(limits=1, user_api='blas'):
        freeze['smoke'] = smoke(models, freeze)
    freeze['smoke_seconds'] = time.perf_counter() - started
    (evidence / 'freeze.json').write_text(json.dumps(freeze, indent=1) + '\n')
    print(json.dumps(freeze['smoke'], indent=1))
    print('freeze sha256:', sha256(evidence / 'freeze.json'))


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    main(Path(sys.argv[1]).resolve())
