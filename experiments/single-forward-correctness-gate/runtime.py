"""Actual mapped-image -> frozen CPU model -> features -> global decisions."""
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from protocol import OLD, ROOT, data, decide, execution, load, module, peak_mib, write_csv
from onboard import scoring, tiles

mlstac = module('single_forward_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')


@torch.no_grad()
def predict(members, x):
    logits = scoring.forward_tiles(members[0], x)
    p = logits.softmax(-1)
    output = {'conf': p.max(-1).values.mean(1).numpy(),
              'pred_cloud_frac': (logits.argmax(-1) == 2).float().mean(1).numpy()}
    if len(members) == 5:
        summed = p.clone()
        for member in members[1:]:
            summed += scoring.forward_tiles(member, x).softmax(-1)
        average = summed / 5
        output['ens'] = (average * average.clamp_min(1e-12).log()).sum(-1).mean(1).numpy()
    return output


def parity(records):
    reference = {data.key(r): r for r in data.read_csv(OLD / 'tiles.csv')}
    deltas = {}
    for field in ('conf', 'pred_cloud_frac', 'ens', 'sat_frac'):
        if field in records[0]:
            diff = max(abs(float(r[field]) - float(reference[r['key']][field])) for r in records)
            assert diff <= (2e-6 if field in ('conf', 'ens') else 0), (field, diff)
            deltas[field] = diff
    assert all((float(r['pred_cloud_frac']) >= .5) ==
               (float(reference[r['key']]['pred_cloud_frac']) >= .5) for r in records)
    return deltas


def run(task, arm, repetition, smoke=False):
    assert arm in ('single', 'ensemble')
    evidence = task / 'evidence'
    started = time.perf_counter()
    freeze, heads = load(evidence / 'freeze.json'), load(evidence / 'models.json')
    root = Path(freeze['data_root'])
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
    handles = [m.register_forward_hook(count(i)) for i, m in enumerate(members)]
    records, units, datasets, masks = [], [], {}, {}
    for dataset in ('cloudsen', 'hypso'):
        dataset_started = time.perf_counter()
        local = []
        names = freeze['units'][dataset][:1] if smoke else freeze['units'][dataset]
        totals = {name: 0. for name in ('io_seconds', 'preprocess_seconds', 'inference_seconds', 'collection_seconds')}
        for number, name in enumerate(names):
            t = time.perf_counter()
            if dataset == 'cloudsen':
                a, info = mlstac.read_patch(freeze['cloudsen_file'], length, *table[name], bands=[1, 2, 3, 4])
                assert info['count'] == 15 and tuple(info['shape']) == (512, 512)
                addressed_bytes = int(table[name][1])
                saturation = None
            else:
                path = root / 'derived/hypso1' / f'{name}_toa.npy'
                a = np.load(path, allow_pickle=False)
                addressed_bytes = path.stat().st_size
                saturation = None
                if arm == 'ensemble':
                    path = root / 'derived/hypso1' / f'{name}_sat.npy'
                    saturation = np.load(path, allow_pickle=False)
                    addressed_bytes += path.stat().st_size
            timing = {'io_seconds': time.perf_counter() - t}
            t = time.perf_counter()
            if dataset == 'cloudsen':
                image = mlstac.crop(a, tuple(freeze['padding_side']))
                saturation = (image >= 65535).any(0) if arm == 'ensemble' else None
                image = np.moveaxis(image, 0, -1).astype(np.float32) * np.float32(1e-4)
            else:
                image = a
            corners = tiles.grid(*image.shape[:2])
            x = tiles.cut(image, corners)
            assert x.dtype == np.float32 and x.shape[-1] == 4 and np.isfinite(x).all()
            sat_frac = tiles.cut(saturation, corners).reshape(len(x), -1).mean(1) if saturation is not None else None
            timing['preprocess_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            features = predict(members, x)
            timing['inference_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            for tile in range(len(x)):
                row = {'key': f'{dataset}/{name}/{tile}', 'dataset': dataset, 'unit': name, 'tile': tile,
                       **{k: float(v[tile]) for k, v in features.items()}}
                if sat_frac is not None:
                    row['sat_frac'] = float(sat_frac[tile])
                local.append(row)
            timing['collection_seconds'] = time.perf_counter() - t
            for key in totals:
                totals[key] += timing[key]
            units.append({'dataset': dataset, 'unit': name, 'n_tiles': len(x),
                          'input_bytes_addressed': addressed_bytes, **timing})
            if (number + 1) % 25 == 0 or number + 1 == len(names):
                print({'arm': arm, 'pass': repetition, 'dataset': dataset,
                       'units': number + 1, 'total_units': len(names)}, flush=True)
        t = time.perf_counter()
        if arm == 'single':
            setting = 'cloudsen_A' if dataset == 'cloudsen' else 'hypso'
            mask = decide(local, freeze['scale'], heads[setting])
        else:
            mask = data.cut(np.asarray([r['ens'] for r in local]))
        totals['decision_seconds'] = time.perf_counter() - t
        masks[dataset] = mask
        totals.update(total_seconds=time.perf_counter() - dataset_started, n_tiles=len(local), n_units=len(names))
        datasets[dataset] = totals
        records.extend(local)
    operational_seconds = time.perf_counter() - started
    operational_peak_rss_mib = peak_mib()
    for handle in handles:
        handle.remove()
    assert calls == [len(units)] * len(members), calls
    # The alternate convention is an offline extra, outside primary cost.
    t = time.perf_counter()
    crows = [r for r in records if r['dataset'] == 'cloudsen']
    alternate = decide(crows, freeze['scale'], heads['cloudsen_B']) if arm == 'single' else masks['cloudsen']
    alternate_seconds = time.perf_counter() - t
    cindex, hindex = 0, 0
    for row in records:
        if row['dataset'] == 'cloudsen':
            row['accepted_primary'] = int(masks['cloudsen'][cindex])
            row['accepted_B'] = int(alternate[cindex]); cindex += 1
        else:
            row['accepted_primary'] = int(masks['hypso'][hindex])
            row['accepted_B'] = int(masks['hypso'][hindex]); hindex += 1
    delta = parity(records)
    record = execution(started, arm=arm, repetition=repetition, smoke=smoke,
        operational_seconds=operational_seconds, setup_seconds=setup_seconds,
        operational_peak_rss_mib=operational_peak_rss_mib,
        model_load_seconds=model_load_seconds, alternate_convention_seconds=alternate_seconds,
        datasets=datasets, units=units, model_calls=calls,
        tile_equivalent_forwards=sum(u['n_tiles'] for u in units) * len(members),
        n_members=len(members), main_output_parity=True, maximum_feature_deltas=delta,
        torch_threads=torch.get_num_threads(), numerical_processes=1,
        raw_labels_read=False, truth_converted=False, test_or_reserved_access=False, new_download_bytes=0,
        torch=torch.__version__, platform=platform.platform(), cpu=platform.processor(), device='cpu',
        input_boundary=freeze['input_boundary'])
    record['standalone_cloudsen_A_seconds'] = setup_seconds + model_load_seconds + datasets['cloudsen']['total_seconds']
    write_csv(evidence / f'features-{arm}-{repetition}.csv', records)
    data.dump(evidence / f'runtime-{arm}-{repetition}.json', record)
    print({k: record[k] for k in ('arm', 'repetition', 'operational_seconds', 'peak_rss_mib', 'model_calls', 'maximum_feature_deltas')}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], '--smoke' in sys.argv)
