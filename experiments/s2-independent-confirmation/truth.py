"""Offline test-label conversion after all frozen prediction-only passes finish."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import data, execution, fixed, load, mlstac, write_csv
from onboard import tiles


def main(task):
    started = time.perf_counter()
    freeze, manifest = fixed(task)
    evidence = task / 'evidence'
    assert not (evidence / 'labels.csv').exists(), 'Do not rescore or replace truth.'
    assert all((evidence / f'runtime-{run}.json').exists() for run in freeze['runtime_order'])
    _, length, table = mlstac.header(freeze['cloudsen_file'])
    rows = []
    for number, meta in enumerate(manifest):
        name = meta['datapoint_id']
        a, info = mlstac.read_patch(freeze['cloudsen_file'], length, *table[name], bands=[14])
        assert info['shape'] == [512, 512] and set(np.unique(a).tolist()) <= {0, 1, 2, 3}
        r, c = freeze['padding_side']
        outside = np.ones((512, 512), dtype=bool)
        outside[r:r + 509, c:c + 509] = False
        assert (a[:, outside] == 0).all(), ('unexpected label padding', name)
        label = mlstac.crop(a[0], (r, c))
        corner = tiles.grid(*label.shape)
        cut = tiles.cut(label, corner).reshape(len(corner), -1)
        for tile, pixels in enumerate(cut):
            rows.append({'key': f'cloudsen/{name}/{tile}', 'unit': name,
                         'cluster': meta['roi_id'], 'tile': tile,
                         'thick_frac': float((pixels == 1).mean()),
                         'thin_frac': float((pixels == 2).mean())})
        if (number + 1) % 100 == 0 or number + 1 == len(manifest):
            print({'truth_patches': number + 1, 'total': len(manifest)}, flush=True)
    assert len(rows) == freeze['n_tiles']
    write_csv(evidence / 'labels.csv', rows)
    data.dump(evidence / 'truth-execution.json', execution(started, n_patches=len(manifest),
        n_tiles=len(rows), bands_read=[14], raw_labels_read=True,
        occurs_after_all_prediction_passes=True, inference_calls=0))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
