"""Write the freeze and check the texture path on development imagery. No test image or label is read."""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frozen import (A_WEIGHT, CONFIRMATION, GATES, MAPPING, PREMISE, ROOT, SAT_DN, SAT_SHARE, SAT_SHIFT,
                    VETO_SHIFT, VETO_THRESHOLD, ZERO_LABEL, module, q_multiscale, sat_fraction, sha256)
from onboard import tiles

mlstac = module('supplementary_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')
SMOKE_UNITS = 2  # first development patches in the premise manifest order
INPUTS = {'premise_freeze': PREMISE / 'freeze.json',
          'single_forward_freeze': ROOT / 'tasks/single-forward-correctness-gate/evidence/freeze.json',
          'confirmation_freeze': CONFIRMATION / 'freeze.json', 'confirmation_results': CONFIRMATION / 'results.json',
          'features_single_1': CONFIRMATION / 'features-single-1.csv',
          'features_ensemble_1': CONFIRMATION / 'features-ensemble-1.csv',
          'decisions': CONFIRMATION / 'decisions.csv', 'zero_label_results': ZERO_LABEL / 'results.json'}


def read(path, length, entry, side):
    """The test runtime's read path: bands 1-4 raw DN, crop, scale; texture uses channels 1:4."""
    a, info = mlstac.read_patch(path, length, *entry, bands=[1, 2, 3, 4])
    assert info['count'] == 15 and info['shape'] == [512, 512]
    dn = np.moveaxis(mlstac.crop(a, side), 0, -1)
    corners = tiles.grid(*dn.shape[:2])
    x = tiles.cut(dn.astype(np.float32) * np.float32(1e-4), corners)
    return tiles.cut(dn, corners), x


def smoke(freeze):
    import csv
    manifest = json.loads((PREMISE / 'input-manifest.json').read_text())
    side = tuple(json.loads((ROOT / 'tasks/cloudsen12-room/evidence/download.json').read_text())
                 ['first_patch']['padding_side_row_col_start'])
    assert list(side) == freeze['population']['padding_side']
    _, length, table = mlstac.header(manifest['cloudsen_file'])
    with open(PREMISE / 'tiles.csv', newline='') as f:
        units = set(manifest['cloudsen_units'][:SMOKE_UNITS])
        recorded = [r for r in csv.DictReader(f) if r['dataset'] == 'cloudsen' and r['unit'] in units]
    report = []
    for name in manifest['cloudsen_units'][:SMOKE_UNITS]:
        rows = sorted([r for r in recorded if r['unit'] == name], key=lambda r: int(r['tile']))
        dn, x = read(manifest['cloudsen_file'], length, table[name], side)
        assert len(rows) == len(x) == 49
        main = np.asarray([float(r['pred_cloud_frac']) >= .5 for r in rows])
        q_all = q_multiscale(x[..., 1:4], freeze['prototypes'])
        q_sub = q_multiscale(x[main][..., 1:4], freeze['prototypes'])
        expected = np.asarray([float(r['q_multiscale']) for r in rows])
        assert np.array_equal(q_all, expected), (name, float(np.abs(q_all - expected).max()))
        assert np.array_equal(q_sub, q_all[main]), name
        sat = sat_fraction(dn)
        assert np.array_equal(sat, [float(r['sat_frac']) for r in rows]), name
        report.append({'unit': name, 'tiles': len(x), 'cloud_calls': int(main.sum()),
                       'q_multiscale_exact': True, 'cloud_call_subset_exact': True, 'sat_frac_exact': True})
    return {'development_units': report, 'file': manifest['cloudsen_file'],
            'read_path': 'bands [1,2,3,4] raw DN, crop, float32 x 1e-4, 64-pixel tiles; texture on channels 1:4',
            'test_images_read': 0, 'labels_read': 0}


def main(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    evidence.mkdir(exist_ok=True)
    assert not (evidence / 'freeze.json').exists(), 'Do not replace the freeze.'
    premise = json.loads((PREMISE / 'freeze.json').read_text())
    scale = json.loads(INPUTS['single_forward_freeze'].read_text())['scale']
    population = json.loads(INPUTS['confirmation_freeze'].read_text())
    assert population['scale'] == scale
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'experiments/texture-s2-supplementary'],
                                    text=True, cwd=ROOT).strip()
    freeze = {
        'task': 'tasks/texture-s2-supplementary', 'written_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'code_commit': revision, 'uncommitted_code': dirty.splitlines(),
        'population': {k: population[k] for k in ('dataset_revision', 'cloudsen_file', 'expected_file_bytes', 'n_patches',
                                                  'n_rois', 'n_tiles', 'padding_side', 'models_directory')},
        'scale': scale, 'radii': premise['radii'], 'prototypes': premise['prototypes'],
        'cloud_call_mapping': {'name': MAPPING, **premise['mappings'][MAPPING]},
        'constants': {'zero_label_weight': A_WEIGHT, 'veto_threshold': VETO_THRESHOLD, 'veto_shift': VETO_SHIFT,
                      'saturation_dn': SAT_DN, 'saturation_share': SAT_SHARE, 'saturation_shift': SAT_SHIFT,
                      'cloud_call': 'pred_cloud_frac >= 0.5'},
        'arms': {
            'confidence': f"conf; {GATES['confidence']}",
            'ensemble': f"ens; {GATES['ensemble']}",
            'zero-label': '0.5*z(conf) - 0.5*z(pred_cloud_frac), source scale',
            'veto-zl': 'zero-label - 1e3*[main and P_cc > 0.5]',
            'texture-conf': '(conf - 1e6*sat) - main*P_cc; sat = (sat_frac > 0.1) and not main',
            'matched-control': "texture-conf's accepted counts per predicted class; within each class, "
                               'stable descending conf - 1e6*sat'},
        'primary': {'arm': 'texture-conf', 'convention': 'A'},
        'selection': 'stable descending global top ceil(0.8*N); the same score serves A and B',
        'veto_budget_rule': 'flagged tiles rank below every unflagged tile and the count stays ceil(0.8*N); '
                            'if flags exceed 20% of N, the flagged tiles with the highest zero-label scores are accepted',
        'texture_scope': 'q_multiscale and P_cc are computed only on predicted-cloud tiles of the single forward; '
                         'P_cc is 0 elsewhere',
        'bootstrap': {'draws': 2000, 'seed': 7, 'protocol': 'tasks/zero-label-s2-test: default_rng(7) whole-ROI '
                      'draws, stable-sorted concatenation, recomputed cut'},
        'input_sha256': {k: sha256(p) for k, p in INPUTS.items()}}
    freeze['smoke'] = smoke(freeze)
    freeze['smoke']['seconds'] = time.perf_counter() - started
    (evidence / 'freeze.json').write_text(json.dumps(freeze, indent=1) + '\n')
    print(json.dumps(freeze['smoke'], indent=1))
    print('freeze sha256:', sha256(evidence / 'freeze.json'))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
