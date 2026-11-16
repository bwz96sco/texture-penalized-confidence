"""Freeze source-only scales and label-blind target roles before any fit."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).parent))
from data import FEATURES, OLD, ROOT, dump, make_splits, read_csv, scale_source
from model import GUARD, L2, N_BOOT, TEMPERATURE, smoke


def main():
    task = Path(sys.argv[1]).resolve()
    evidence = task / 'evidence'
    evidence.mkdir(exist_ok=True)
    target = read_csv(OLD / 'tiles.csv')
    source = read_csv(OLD / 'source-features.csv')
    assert len(target) == 30135 and len(source) == 3600
    dump(evidence / 'splits.json', make_splits(target))
    dump(evidence / 'freeze.json', {
        'contract_revision': 1,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'features': list(FEATURES), 'scale': scale_source(source), 'l2': L2,
        'temperature': TEMPERATURE, 'n_bootstrap_heads': N_BOOT, 'guard_multiplier': GUARD,
        'n_paired_evaluation_draws': 2000, 'evaluation_seed': 7,
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'labels_accessed': False, 'primary': 'soft-budget-v2',
        'budget': 'one global ceil(0.8*N) per evaluation population',
        'optimizer': {'method': 'L-BFGS-B', 'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 100}})
    dump(evidence / 'smoke.json', smoke())
    print(json.dumps({'n_target': len(target), 'n_source': len(source), 'status': 'prepared_label_blind'}))


if __name__ == '__main__':
    main()
