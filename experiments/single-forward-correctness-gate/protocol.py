"""Fixed single-forward candidate and retained commissioning/evaluation roles."""
import csv
import datetime as dt
import importlib.util
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'tasks/net-risk-calibrated-swaps/evidence'
OLD = ROOT / 'tasks/texture-permutation-premise/evidence'
FEATURES = ('conf', 'pred_cloud_frac')
DRAWS = 2000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


data = module('single_forward_data', ROOT / 'experiments/budget-swap-gate/data.py')
sys.modules['data'] = data
legacy = module('single_forward_fit', ROOT / 'experiments/budget-swap-gate/model.py')
SETTINGS = data.SETTINGS


def load(path):
    return json.loads(Path(path).read_text())


def take(rows, keys):
    lookup = {data.key(r): r for r in rows}
    assert len(lookup) == len(rows)
    return [lookup[k] for k in keys]


def matrix(rows, scale):
    return (data.values(rows, FEATURES) - np.asarray(scale['mean'])) / np.asarray(scale['sd'])


def fit_heads(rows, splits, scale):
    heads = {}
    for name, dataset, convention in SETTINGS:
        ids = splits['datasets'][dataset]['ordinary_training_keys']
        training = take(rows, ids)
        assert len(ids) == len(set(ids)) == 128
        assert set(data.group(r) for r in training).isdisjoint(splits['datasets'][dataset]['evaluation_groups'])
        error, _ = data.errors(training, convention)
        x = matrix(training, scale)
        heads[name] = legacy.fit('pointwise', np.zeros(len(training)), x, error)
        heads[name]['training_keys'] = ids
    return heads


def decide(rows, scale, head):
    # Only the two declared columns are ever read; no offset or prior mask.
    score = matrix(rows, scale) @ np.asarray(head['theta']) + head['intercept']
    return data.cut(score)


def write_csv(path, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if platform.system() == 'Darwin' else 1024)


def execution(started, **extra):
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'command': ['uv', 'run', '--no-sync', 'python', '-I', *sys.argv],
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'total_seconds': time.perf_counter() - started, 'peak_rss_mib': peak_mib(), **extra}
