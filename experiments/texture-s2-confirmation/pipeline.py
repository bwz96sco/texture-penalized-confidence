"""Population-agnostic per-patch features for the frozen texture configuration.

One read of bands 1-4 per patch: single forward (member 0) and the five-member ensemble,
saturation share, and texture with the cloud-call mapping on predicted-cloud tiles only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/texture-s2-supplementary'))
from frozen import module, p_cloud_call, q_multiscale, sat_fraction  # noqa: E402
from onboard import scoring, tiles  # noqa: E402

predict = module('confirmation_texture_predict', ROOT / 'experiments/single-forward-correctness-gate/runtime.py').predict
mlstac = module('confirmation_texture_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')


def members(directory):
    expected = json.loads((ROOT / 'tasks/premise-test/evidence/freeze.json').read_text())['model']['member_sha256']
    loaded = []
    for i in range(5):
        path = Path(directory) / f'member{i}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[str(i)], path
        loaded.append(scoring.load_member(str(path)))
    return loaded


def patch(models, path, length, entry, side, prototypes, mapping):
    """Per-tile features of one 512 x 512 patch; the label band is never requested."""
    a, info = mlstac.read_patch(path, length, *entry, bands=[1, 2, 3, 4])
    assert info['count'] == 15 and info['shape'] == [512, 512]
    r, c = side
    outside = np.ones((512, 512), dtype=bool)
    outside[r:r + 509, c:c + 509] = False
    assert (a[:, outside] == 0).all(), 'unexpected image padding'
    dn = np.moveaxis(mlstac.crop(a, (r, c)), 0, -1)
    corners = tiles.grid(*dn.shape[:2])
    x = tiles.cut(dn.astype(np.float32) * np.float32(1e-4), corners)
    assert len(x) == 49 and np.isfinite(x).all()
    out = predict(models, x)
    main = out['pred_cloud_frac'] >= .5
    q = np.full(len(x), np.nan)
    q[main] = q_multiscale(x[main][..., 1:4], prototypes)
    p = np.zeros(len(x))
    p[main] = p_cloud_call(q[main], mapping)
    sat = sat_fraction(tiles.cut(dn, corners))
    return [{'tile': t, 'conf': float(out['conf'][t]), 'pred_cloud_frac': float(out['pred_cloud_frac'][t]),
             'ens': float(out['ens'][t]), 'sat_frac': float(sat[t]), 'q_multiscale': float(q[t]), 'p_cc': float(p[t])}
            for t in range(len(x))]
