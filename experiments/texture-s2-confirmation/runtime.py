"""One image-only prediction pass over the untouched sample; the label band is never requested."""
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ROOT, members, mlstac, patch  # noqa: E402
from frozen import sha256  # noqa: E402

FREEZE_SHA256 = '8bfde43607cfdadc6c38fe1317264df054e85477898412d3ee68a18d0aeaecf7'


def run(task):
    evidence = task / 'evidence'
    assert FREEZE_SHA256 and sha256(evidence / 'freeze.json') == FREEZE_SHA256, 'freeze differs from the contract'
    assert not (evidence / 'features.csv').exists(), 'One prediction pass only.'
    started = time.perf_counter()
    freeze = json.loads((evidence / 'freeze.json').read_text())
    population = freeze['population']
    manifest_path = ROOT / population['manifest']
    assert sha256(manifest_path) == population['manifest_sha256']
    with open(manifest_path, newline='') as f:
        manifest = list(csv.DictReader(f))
    assert len(manifest) == population['n_patches']
    magic, length, table = mlstac.header(population['file'])
    assert magic == b'#y'
    models = members(population['models_directory'])
    records = []
    for number, meta in enumerate(manifest):
        name = meta['datapoint_id']
        assert table[name] == [int(meta['begin']), int(meta['length'])]
        for row in patch(models, population['file'], length, table[name], population['padding_side'],
                         freeze['prototypes'], freeze['cloud_call_mapping']):
            records.append({'key': f"cloudsen/{name}/{row['tile']}", 'unit': name, 'cluster': meta['roi_id'],
                            **{k: repr(v) for k, v in row.items() if k != 'p_cc'},
                            'q_multiscale': repr(row['q_multiscale']) if row['pred_cloud_frac'] >= .5 else ''})
        if (number + 1) % 200 == 0 or number + 1 == len(manifest):
            print({'patches': number + 1, 'total': len(manifest), 'seconds': time.perf_counter() - started}, flush=True)
    assert len(records) == population['n_tiles']
    with open(evidence / 'features.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (evidence / 'runtime.json').write_text(json.dumps({
        'seconds': time.perf_counter() - started, 'freeze_sha256': FREEZE_SHA256, 'n_patches': len(manifest),
        'n_tiles': len(records), 'members': 5, 'torch_threads': torch.get_num_threads(), 'blas_threads': 1,
        'platform': platform.platform(), 'torch': torch.__version__, 'imagery_bands_read': [1, 2, 3, 4],
        'label_band_requested': False, 'raw_labels_read': False}, indent=1) + '\n')


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(Path(sys.argv[1]).resolve())
