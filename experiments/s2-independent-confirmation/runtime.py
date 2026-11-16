"""Image-only complete test replay; raw labels are never requested here."""
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, data, decide, execution, fixed, mlstac, module, peak_mib, verify_weights, write_csv
from onboard import scoring, tiles

predict = module('confirmation_frozen_predict', ROOT / 'experiments/single-forward-correctness-gate/runtime.py').predict


def run(task, arm, repetition, smoke=False):
    assert arm in ('single', 'ensemble')
    freeze, _ = fixed(task)
    verify_weights(freeze)
    runner_started = time.perf_counter()
    started = time.perf_counter()
    freeze, manifest = fixed(task)
    magic, length, table = mlstac.header(freeze['cloudsen_file'])
    assert magic == b'#y'
    setup_seconds = time.perf_counter() - started
    t = time.perf_counter()
    members = [scoring.load_member(str(Path(freeze['models_directory']) / f'member{i}.pt'))
               for i in range(1 if arm == 'single' else 5)]
    model_load_seconds = time.perf_counter() - t
    calls = [0] * len(members)
    def count(index):
        def hook(*_):
            calls[index] += 1
        return hook
    handles = [member.register_forward_hook(count(i)) for i, member in enumerate(members)]
    selected = manifest[:1] if smoke else manifest
    records, units = [], []
    totals = {k: 0.0 for k in ('io_seconds', 'preprocess_seconds', 'inference_seconds', 'collection_seconds')}
    pipeline_started = time.perf_counter()
    for number, meta in enumerate(selected):
        name = meta['datapoint_id']
        t = time.perf_counter()
        a, info = mlstac.read_patch(freeze['cloudsen_file'], length, *table[name], bands=[1, 2, 3, 4])
        assert info['count'] == 15 and info['shape'] == [512, 512]
        timing = {'io_seconds': time.perf_counter() - t}
        t = time.perf_counter()
        r, c = freeze['padding_side']
        outside = np.ones((512, 512), dtype=bool)
        outside[r:r + 509, c:c + 509] = False
        assert (a[:, outside] == 0).all(), ('unexpected image padding', name)
        image = np.moveaxis(mlstac.crop(a, (r, c)), 0, -1).astype(np.float32) * np.float32(1e-4)
        corners = tiles.grid(*image.shape[:2])
        x = tiles.cut(image, corners)
        assert len(x) == 49 and x.dtype == np.float32 and np.isfinite(x).all()
        timing['preprocess_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        features = predict(members, x)
        timing['inference_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        for tile in range(len(x)):
            records.append({'key': f'cloudsen/{name}/{tile}', 'dataset': 'cloudsen',
                            'unit': name, 'cluster': meta['roi_id'], 'tile': tile,
                            **{k: float(v[tile]) for k, v in features.items()}})
        timing['collection_seconds'] = time.perf_counter() - t
        for key in totals:
            totals[key] += timing[key]
        units.append({'unit': name, 'n_tiles': len(x), 'input_bytes_addressed': table[name][1], **timing})
        if (number + 1) % 100 == 0 or number + 1 == len(selected):
            print({'arm': arm, 'pass': repetition, 'patches': number + 1, 'total': len(selected)}, flush=True)
    t = time.perf_counter()
    accepted = decide(records, freeze['scale'], freeze['heads']['cloudsen_A']) if arm == 'single' else data.cut(np.asarray([r['ens'] for r in records]))
    totals['decision_seconds'] = time.perf_counter() - t
    totals['total_seconds'] = time.perf_counter() - pipeline_started
    operational_seconds = time.perf_counter() - started
    rss = peak_mib()
    for handle in handles:
        handle.remove()
    assert calls == [len(selected)] * len(members)
    t = time.perf_counter()
    alternate = decide(records, freeze['scale'], freeze['heads']['cloudsen_B']) if arm == 'single' else accepted
    alternate_seconds = time.perf_counter() - t
    for i, row in enumerate(records):
        row['accepted_primary'] = int(accepted[i])
        row['accepted_B'] = int(alternate[i])
    record = execution(runner_started, arm=arm, repetition=repetition, smoke=smoke,
        operational_seconds=operational_seconds, operational_peak_rss_mib=rss,
        standalone_cloudsen_A_seconds=operational_seconds, setup_seconds=setup_seconds,
        model_load_seconds=model_load_seconds, pipeline=totals, units=units,
        alternate_convention_seconds=alternate_seconds, n_members=len(members),
        model_calls=calls, tile_equivalent_forwards=len(records) * len(members),
        torch_threads=torch.get_num_threads(), numerical_processes=1,
        raw_labels_read=False, truth_converted=False, imagery_bands_read=[1, 2, 3, 4],
        test_access_authorized=True, reserved_hypso_access=False,
        new_download_bytes=0, torch=torch.__version__, platform=platform.platform(),
        device='cpu', input_boundary=freeze['input_boundary'])
    evidence = task / 'evidence'
    write_csv(evidence / f'features-{arm}-{repetition}.csv', records)
    data.dump(evidence / f'runtime-{arm}-{repetition}.json', record)
    print({'arm': arm, 'pass': repetition, 'seconds': operational_seconds,
           'peak_rss_mib': rss, 'calls': calls}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], '--smoke' in sys.argv)
