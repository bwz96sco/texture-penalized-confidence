"""Label conversion for the untouched sample, only after the prediction pass has finished."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT, mlstac  # noqa: E402
from onboard import tiles  # noqa: E402


def main(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    assert (evidence / 'features.csv').exists() and (evidence / 'runtime.json').exists(), 'Predict first.'
    assert not (evidence / 'labels.csv').exists(), 'Do not replace truth.'
    freeze = json.loads((evidence / 'freeze.json').read_text())
    population = freeze['population']
    with open(ROOT / population['manifest'], newline='') as f:
        manifest = list(csv.DictReader(f))
    _, length, table = mlstac.header(population['file'])
    r, c = population['padding_side']
    rows = []
    for meta in manifest:
        name = meta['datapoint_id']
        a, info = mlstac.read_patch(population['file'], length, *table[name], bands=[14])
        assert info['shape'] == [512, 512] and set(np.unique(a).tolist()) <= {0, 1, 2, 3}, name
        outside = np.ones((512, 512), dtype=bool)
        outside[r:r + 509, c:c + 509] = False
        assert (a[:, outside] == 0).all(), ('unexpected label padding', name)
        label = mlstac.crop(a[0], (r, c))
        cut = tiles.cut(label, tiles.grid(*label.shape)).reshape(49, -1)
        for tile, pixels in enumerate(cut):
            rows.append({'key': f'cloudsen/{name}/{tile}', 'cluster': meta['roi_id'],
                         'thick_frac': repr(float((pixels == 1).mean())), 'thin_frac': repr(float((pixels == 2).mean()))})
    assert len(rows) == population['n_tiles']
    with open(evidence / 'labels.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (evidence / 'truth.json').write_text(json.dumps({'seconds': time.perf_counter() - started, 'n_patches': len(manifest),
                                                     'n_tiles': len(rows), 'band_read': 14,
                                                     'after_prediction_pass': True}, indent=1) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
