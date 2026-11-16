"""Reuse the closed candidate, predictor and global-count evaluator without refitting."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/single-forward-correctness-gate'))
from protocol import data, decide, execution, load, module, peak_mib, write_csv

PRIOR = ROOT / 'tasks/single-forward-correctness-gate/evidence'
REVISION = '2bcd2e9d1a16f8986e8eed6df893f92cd5bca155'
DATA_ROOT = Path.home() / 'Data/onboard-acceptance/cloudsen12plus-test-confirmation-20261017'
FILE_NAME = 'test_509_high.mlstac'
URL = f'https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus/resolve/{REVISION}/test/{FILE_NAME}'
GATES = {'confidence': 'softmax-response-cloudsen/v1', 'ensemble': 'ensemble-entropy-cloudsen/v1'}
mlstac = module('confirmation_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')


def fixed(task):
    freeze = load(task / 'evidence/freeze.json')
    old = load(PRIOR / 'freeze.json')
    assert freeze['scale'] == old['scale']
    assert freeze['heads'] == {k: v for k, v in load(PRIOR / 'models.json').items() if k.startswith('cloudsen_')}
    assert freeze['models_directory'] == old['models_directory']
    assert freeze['padding_side'] == old['padding_side']
    return freeze, data.read_csv(task / 'evidence/manifest.csv')


def verify_weights(freeze):
    expected = load(ROOT / 'tasks/premise-test/evidence/freeze.json')['model']['member_sha256']
    for member, digest in expected.items():
        path = Path(freeze['models_directory']) / f'member{member}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def scores(rows, freeze, convention):
    x = (data.values(rows, ('conf', 'pred_cloud_frac')) - np.asarray(freeze['scale']['mean'])) / np.asarray(freeze['scale']['sd'])
    head = freeze['heads'][f'cloudsen_{convention}']
    return {'single': x @ np.asarray(head['theta']) + head['intercept'],
            'confidence': np.asarray([float(r['conf']) for r in rows]),
            'ensemble': np.asarray([float(r['ens']) for r in rows])}


def masks(rows, freeze, convention):
    return {k: data.cut(v) for k, v in scores(rows, freeze, convention).items()}
