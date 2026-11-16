"""Fix the untouched CloudSEN12+ train-509-high sample from metadata only.

Only identifiers, split, shape, label type, ROI, centroid, date, URL and file offsets
are used. Label-percentage, `fixed` and `abs_offset` fields are dropped on read and
never written.
"""
import csv
import glob
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HOME = Path.home() / 'Data/onboard-acceptance'
DATA = HOME / 'cloudsen12plus-train509-confirmation-20261019'
TRAINVAL = HOME / 'meta-cloudsen12plus/cloudsen12plus_high_trainval_metadata.csv'
TEST_PAGES = HOME / 'cloudsen12plus-test-confirmation-20261017'
REVISION = '2bcd2e9d1a16f8986e8eed6df893f92cd5bca155'
FILE = 'train/train_509_high.mlstac'
FIELDS = ('datapoint_id', 'roi_id', 'split', 'proj_shape', 'label_type', 'proj_centroid', 's2_date',
          'url', 'metadata_length', 'begin', 'length')
SEED, N_ROIS, RADIUS_KM = 20261019, 330, 10.0


def project(row):
    return {k: str(row[k]) for k in FIELDS}


def validation_rows():
    """All validation rows (every shape and label type) from the datasets server, cached in DATA."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        path = DATA / f'validation-metadata-50-{offset:04d}.json'
        if not path.exists():
            query = urllib.parse.urlencode(dict(dataset='isp-uv-es/CloudSEN12Plus', config='default',
                                                split='validation', offset=offset, length=50))
            for attempt in range(3):
                try:
                    with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?' + query, timeout=60) as r:
                        chunks = []
                        while chunk := r.read(65536):
                            chunks.append(chunk)
                    payload = b''.join(chunks)
                    json.loads(payload)
                    path.write_bytes(payload)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10)
        page = json.loads(path.read_text())
        total = page['num_rows_total']
        rows += [project(item['row']) for item in page['rows']]
        offset += 50
    assert len(rows) == total and len({r['datapoint_id'] for r in rows}) == total
    return rows


def test_rows():
    rows = []
    for path in sorted(glob.glob(str(TEST_PAGES / 'test-metadata-50-*.json'))):
        rows += [project(item['row']) for item in json.loads(Path(path).read_text())['rows']]
    assert len(rows) == len({r['datapoint_id'] for r in rows}) == 1715
    return rows


def lonlat(point):
    lon, lat = point.removeprefix('POINT (').removesuffix(')').split()
    return math.radians(float(lon)), math.radians(float(lat))


def km(a, b):
    (l1, p1), (l2, p2) = a, b
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def select():
    with open(TRAINVAL, newline='') as f:
        trainval = [project(r) for r in csv.DictReader(f)]
    validation, test = validation_rows(), test_rows()
    train = [r for r in trainval if r['split'] == 'train' and r['proj_shape'] == '509' and r['label_type'] == 'high']
    assert len(train) == 8490 and all(r['url'].endswith('/' + FILE) for r in train)
    not_train = {r['roi_id'] for r in validation + test}
    used = [r for r in validation + test if r['proj_shape'] == '509' and r['label_type'] == 'high']
    used_points = {r['roi_id']: lonlat(r['proj_centroid']) for r in used}
    by_roi = {}
    for r in train:
        by_roi.setdefault(r['roi_id'], []).append(r)
    excluded_split, excluded_near, eligible, nearest = [], [], [], {}
    for roi in sorted(by_roi):
        if roi in not_train:
            excluded_split.append(roi)
            continue
        point = lonlat(by_roi[roi][0]['proj_centroid'])
        nearest[roi] = min(km(point, p) for p in used_points.values())
        (excluded_near if nearest[roi] <= RADIUS_KM else eligible).append(roi)
    picked = np.random.default_rng(SEED).choice(len(eligible), size=N_ROIS, replace=False)
    sample = sorted(eligible[int(i)] for i in picked)
    manifest = sorted((r for roi in sample for r in by_roi[roi]), key=lambda r: r['datapoint_id'])
    selection = {
        'source': {'revision': REVISION, 'file': FILE, 'trainval_metadata': str(TRAINVAL),
                   'validation_pages': len(list(DATA.glob('validation-metadata-50-*.json'))),
                   'test_pages': str(TEST_PAGES)},
        'fields_used': list(FIELDS), 'label_derived_fields_used': False,
        'rule': {'eligible': 'train-509-high ROIs not in any validation or test row and more than '
                             f'{RADIUS_KM} km (great circle) from every validation-509-high or test-509-high ROI centroid',
                 'sample': f'numpy default_rng({SEED}).choice(len(sorted eligible), {N_ROIS}, replace=False); '
                           'all patches of each sampled ROI'},
        'counts': {'train_509_high_patches': len(train), 'train_509_high_rois': len(by_roi),
                   'validation_rows': len(validation), 'test_rows': len(test), 'used_rois': len(used_points),
                   'excluded_in_other_split': len(excluded_split), 'excluded_within_radius': len(excluded_near),
                   'eligible_rois': len(eligible), 'sampled_rois': len(sample), 'manifest_patches': len(manifest),
                   'manifest_tiles': 49 * len(manifest), 'reserve_rois': len(eligible) - len(sample)},
        'nearest_used_km_sampled': {'min': min(nearest[r] for r in sample), 'median': float(np.median([nearest[r] for r in sample]))},
        'excluded_within_radius': excluded_near, 'eligible_rois': eligible, 'sampled_rois': sample,
        'reserve_rois': sorted(set(eligible) - set(sample))}
    return selection, manifest


def main(task):
    evidence = task / 'evidence'
    evidence.mkdir(exist_ok=True)
    assert not (evidence / 'manifest.csv').exists(), 'Do not replace the fixed sample.'
    selection, manifest = select()
    with open(evidence / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(manifest)
    (evidence / 'selection.json').write_text(json.dumps(selection, indent=1) + '\n')
    print(json.dumps({k: selection[k] for k in ('counts', 'nearest_used_km_sampled')}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
