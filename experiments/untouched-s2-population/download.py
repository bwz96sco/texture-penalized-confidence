"""Fetch the header and the sampled patches of the pinned train-509-high file into a sparse local copy.

The header table is checked against the metadata before any patch range is fetched.
Patches are opened header-only; no band's pixel values are read.
"""
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rule = load('untouched_select', Path(__file__).parent / 'select.py')  # 'select' clashes with the stdlib module
DATA, FILE, REVISION, TRAINVAL, project = rule.DATA, rule.FILE, rule.REVISION, rule.TRAINVAL, rule.project
URL = f'https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus/resolve/{REVISION}/{FILE}'
TARGET = DATA / 'train_509_high.mlstac'


def mlstac():
    return load('untouched_mlstac', ROOT / 'experiments/cloudsen12-room/mlstac.py')


def fetch(start, end, name, total):
    """One HTTP range into DATA/ranges/name; returns its record after checking 206, content-range and size."""
    directory = DATA / 'ranges'
    path, headers = directory / name, directory / f'{name}.headers'
    argv = ['curl', '--location', '--fail', '--silent', '--show-error', '--connect-timeout', '30',
            '--max-time', '1800', '--range', f'{start}-{end}', '--dump-header', str(headers),
            '--output', str(path), '--write-out',
            '{"http_code":%{http_code},"size_download":%{size_download},"time_total":%{time_total}}', URL]
    for attempt in range(4):
        run = subprocess.run(argv, capture_output=True, text=True)
        ok = (run.returncode == 0 and path.exists() and path.stat().st_size == end - start + 1
              and f'content-range: bytes {start}-{end}/{total}' in headers.read_text().lower())
        if ok:
            return {'name': name, 'start': start, 'end': end, 'bytes': end - start + 1,
                    'attempts': attempt + 1, **json.loads(run.stdout)}
        time.sleep(15 * (attempt + 1))
    raise RuntimeError((name, start, end, run.returncode, run.stderr))


def place(record):
    path = DATA / 'ranges' / record['name']
    with open(TARGET, 'r+b') as dst, open(path, 'rb') as src:
        dst.seek(record['start'])
        dst.write(src.read())
    path.unlink()
    (DATA / 'ranges' / f"{record['name']}.headers").unlink()


def main(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    assert not (evidence / 'download.json').exists(), 'Transfer already recorded.'
    tree = {x['path']: x for x in json.loads((DATA / 'repository-train-tree.json').read_text())}[FILE]
    total = tree['size']
    with open(TRAINVAL, newline='') as f:
        rows = [project(r) for r in csv.DictReader(f)]
    rows = [r for r in rows if r['url'].endswith('/' + FILE)]
    assert len(rows) == 8490
    with open(evidence / 'manifest.csv', newline='') as f:
        manifest = list(csv.DictReader(f))
    length = int(rows[0]['metadata_length'])
    (DATA / 'ranges').mkdir(exist_ok=True)
    if not TARGET.exists():
        with open(TARGET, 'wb') as f:
            f.truncate(total)  # sparse on APFS: only fetched ranges occupy disk
    header = fetch(0, 10 + length - 1, 'header', total)
    place(header)
    m = mlstac()
    checks, length_read, table = m.check(TARGET, rows)
    assert checks['magic_ok'] and checks['size_ok'] and checks['header_length_matches_metadata'] \
        and checks['table_matches_metadata'] and length_read == length, checks
    spans = []
    for r in sorted(manifest, key=lambda r: int(r['begin'])):
        a = 10 + length + int(r['begin'])
        b = a + int(r['length']) - 1
        if spans and spans[-1][1] + 1 == a:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda s: fetch(s[1][0], s[1][1], f'span-{s[0]:04d}', total), enumerate(spans)))
    for record in records:
        place(record)
    import rasterio
    opened = []
    for r in manifest:
        begin, nbytes = table[r['datapoint_id']]
        with rasterio.open(f'/vsisubfile/{10 + length + begin}_{nbytes},{TARGET}') as src:
            opened.append({'count': src.count, 'shape': [src.height, src.width], 'dtypes': sorted(set(src.dtypes))})
    assert all(o['count'] == 15 and o['shape'] == [512, 512] for o in opened)
    disk = int(subprocess.check_output(['du', '-k', str(TARGET)], text=True).split()[0]) * 1024
    record = {'url': URL, 'revision': REVISION, 'file': FILE, 'tree_size': total, 'lfs_oid': tree['lfs']['oid'],
              'header_checks': checks, 'header_bytes': header['bytes'], 'patches': len(manifest), 'spans': len(spans),
              'patch_bytes': sum(x['bytes'] for x in records), 'retries': sum(x['attempts'] - 1 for x in records),
              'transfer_seconds_sum': sum(x['time_total'] for x in records),
              'all_ranges_206_content_range_and_size_verified': True,
              'patches_open_15_bands_512': True, 'dtypes': sorted({d for o in opened for d in o['dtypes']}),
              'sparse_file_apparent_bytes': TARGET.stat().st_size, 'sparse_file_disk_bytes': disk,
              'pixel_values_read': False, 'label_band_read': False, 'model_run': False,
              'local_path': str(TARGET), 'wall_seconds': time.perf_counter() - started, 'spans_detail': records}
    (evidence / 'download.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'spans_detail'}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
