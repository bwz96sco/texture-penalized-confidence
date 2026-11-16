"""Image-only timed pipeline on the 165 independent S2 test ROIs; raw labels are never requested.

Per patch: read bands 1-4 -> crop/scale/tile -> single forward -> saturation share ->
texture and cloud-call mapping on predicted-cloud tiles only. Then the global
texture-conf decision. A reference texture pass over all tiles is timed separately,
outside the pipeline, to measure what restricting texture to cloud calls saves.
"""
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from frozen import CONFIRMATION, ROOT, SAT_SHARE, SAT_SHIFT, cut, module, p_cloud_call, q_multiscale, sat_fraction, sha256
from onboard import scoring, tiles

protocol = module('supplementary_protocol', ROOT / 'experiments/single-forward-correctness-gate/protocol.py')
predict = module('supplementary_predict', ROOT / 'experiments/single-forward-correctness-gate/runtime.py').predict
mlstac = module('supplementary_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')
FREEZE_SHA256 = 'f9014a759b799260e4f67bddc5b94edc7329f48c277ae6b818ec7b821df27d59'
STEPS = ('io_seconds', 'preprocess_seconds', 'inference_seconds', 'saturation_seconds', 'texture_seconds',
         'mapping_seconds', 'collection_seconds')


def hardware():
    sysctl = lambda key: subprocess.check_output(['sysctl', '-n', key], text=True).strip()
    return {'cpu': sysctl('machdep.cpu.brand_string'), 'logical_cpus': int(sysctl('hw.ncpu')),
            'memory_bytes': int(sysctl('hw.memsize')), 'platform': platform.platform(), 'device': 'cpu',
            'torch_threads': torch.get_num_threads(), 'blas_threads': 1, 'numerical_processes': 1}


def run(task, repetition):
    evidence = task / 'evidence'
    assert sha256(evidence / 'freeze.json') == FREEZE_SHA256, 'freeze changed after the contract was retained'
    runner_started = time.perf_counter()
    started = time.perf_counter()
    freeze = json.loads((evidence / 'freeze.json').read_text())
    population = freeze['population']
    with open(CONFIRMATION / 'manifest.csv', newline='') as f:
        manifest = list(csv.DictReader(f))
    assert len(manifest) == population['n_patches']
    magic, length, table = mlstac.header(population['cloudsen_file'])
    assert magic == b'#y'
    expected = json.loads((ROOT / 'tasks/premise-test/evidence/freeze.json').read_text())['model']['member_sha256']['0']
    weights = Path(population['models_directory']) / 'member0.pt'
    assert sha256(weights) == expected, weights
    setup_seconds = time.perf_counter() - started
    t = time.perf_counter()
    member = scoring.load_member(str(weights))
    model_load_seconds = time.perf_counter() - t
    prototypes, mapping, side = freeze['prototypes'], freeze['cloud_call_mapping'], population['padding_side']
    records, totals = [], {k: 0.0 for k in STEPS}
    reference_texture_seconds, reference_equal = 0.0, True
    pipeline_started = time.perf_counter()
    for number, meta in enumerate(manifest):
        name, timing = meta['datapoint_id'], {}
        t = time.perf_counter()
        a, info = mlstac.read_patch(population['cloudsen_file'], length, *table[name], bands=[1, 2, 3, 4])
        assert info['count'] == 15 and info['shape'] == [512, 512]
        timing['io_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        r, c = side
        outside = np.ones((512, 512), dtype=bool)
        outside[r:r + 509, c:c + 509] = False
        assert (a[:, outside] == 0).all(), ('unexpected image padding', name)
        dn = np.moveaxis(mlstac.crop(a, (r, c)), 0, -1)
        corners = tiles.grid(*dn.shape[:2])
        x = tiles.cut(dn.astype(np.float32) * np.float32(1e-4), corners)
        assert len(x) == 49 and x.dtype == np.float32 and np.isfinite(x).all()
        timing['preprocess_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        out = predict([member], x)
        timing['inference_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        sat = sat_fraction(tiles.cut(dn, corners))
        timing['saturation_seconds'] = time.perf_counter() - t
        main = out['pred_cloud_frac'] >= .5
        t = time.perf_counter()
        q = q_multiscale(x[main][..., 1:4], prototypes)
        timing['texture_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        p = p_cloud_call(q, mapping)
        timing['mapping_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        q_full, p_full = np.full(len(x), np.nan), np.zeros(len(x))
        q_full[main], p_full[main] = q, p
        for tile in range(len(x)):
            records.append({'key': f'cloudsen/{name}/{tile}', 'conf': float(out['conf'][tile]),
                            'pred_cloud_frac': float(out['pred_cloud_frac'][tile]), 'sat_frac': float(sat[tile]),
                            'q_multiscale': float(q_full[tile]), 'p_cc': float(p_full[tile])})
        timing['collection_seconds'] = time.perf_counter() - t
        for key in STEPS:
            totals[key] += timing[key]
        t = time.perf_counter()  # reference only: texture on every tile, outside the pipeline
        q_all = q_multiscale(x[..., 1:4], prototypes)
        reference = time.perf_counter() - t
        reference_texture_seconds += reference
        pipeline_started += reference  # keep the reference pass out of the pipeline total
        reference_equal &= bool(np.array_equal(q_all[main], q))
        if (number + 1) % 100 == 0 or number + 1 == len(manifest):
            print({'pass': repetition, 'patches': number + 1, 'total': len(manifest)}, flush=True)
    t = time.perf_counter()
    conf = np.asarray([r['conf'] for r in records])
    main = np.asarray([r['pred_cloud_frac'] >= .5 for r in records])
    sat = (np.asarray([r['sat_frac'] for r in records]) > SAT_SHARE) & ~main
    accepted = cut((conf - SAT_SHIFT * sat) - main * np.asarray([r['p_cc'] for r in records]))
    totals['decision_seconds'] = time.perf_counter() - t
    totals['total_seconds'] = time.perf_counter() - pipeline_started
    operational_seconds = time.perf_counter() - started - reference_texture_seconds
    rss = protocol.peak_mib()
    assert reference_equal, 'cloud-call texture differs from the all-tile reference'
    retained = protocol.data.read_csv(CONFIRMATION / 'features-single-1.csv')
    assert [r['key'] for r in retained] == [r['key'] for r in records]
    deltas = {k: max(abs(float(o[k]) - n[k]) for o, n in zip(retained, records)) for k in ('conf', 'pred_cloud_frac')}
    assert deltas['conf'] <= 2e-6 and deltas['pred_cloud_frac'] == 0, deltas
    rows = [{'key': row['key'], 'sat_frac': repr(row['sat_frac']),
             'q_multiscale': repr(row['q_multiscale']) if m else '', 'accepted_texture_conf': str(int(a))}
            for row, m, a in zip(records, main, accepted)]
    features = evidence / 'features.csv'
    if repetition == '1':
        assert not features.exists(), 'Do not replace the first pass.'
        with open(features, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    else:
        with open(features, newline='') as f:
            assert list(csv.DictReader(f)) == rows, 'repeat pass differs from the first'
    record = protocol.execution(
        runner_started, repetition=repetition, freeze_sha256=FREEZE_SHA256,
        operational_seconds=operational_seconds, operational_peak_rss_mib=rss, setup_seconds=setup_seconds,
        model_load_seconds=model_load_seconds, pipeline=totals,
        reference_all_tile_texture_seconds=reference_texture_seconds,
        n_patches=len(manifest), n_tiles=len(records), n_cloud_calls_textured=int(main.sum()),
        n_saturation_flags=int(sat.sum()), n_accepted=int(accepted.sum()),
        cloud_call_texture_equals_all_tile_reference=reference_equal, parity_with_retained_features=deltas,
        hardware=hardware(), torch=torch.__version__, raw_labels_read=False, label_band_requested=False,
        imagery_bands_read=[1, 2, 3, 4], reserved_hypso_access=False, new_download_bytes=0)
    (evidence / f'runtime-{repetition}.json').write_text(json.dumps(record, indent=1) + '\n')
    print({'pass': repetition, 'seconds': operational_seconds, 'pipeline': totals,
           'reference_all_tile_texture_seconds': reference_texture_seconds, 'cloud_calls': int(main.sum()),
           'saturation_flags': int(sat.sum()), 'parity': deltas}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(Path(sys.argv[1]).resolve(), sys.argv[2])
