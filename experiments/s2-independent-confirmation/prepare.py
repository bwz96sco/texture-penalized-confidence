"""Fix test IDs from metadata, then check the downloaded container without labels."""
import csv
import shutil
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_ROOT, FILE_NAME, PRIOR, REVISION, ROOT, URL, data, execution, load, mlstac, write_csv

FIELDS = ('datapoint_id', 'roi_id', 'split', 'proj_shape', 'label_type', 'url', 'metadata_length', 'begin', 'length')


def page(offset):
    path = DATA_ROOT / f'test-metadata-50-{offset:04d}.json'
    query = urllib.parse.urlencode(dict(dataset='isp-uv-es/CloudSEN12Plus', config='default', split='test', offset=offset, length=50))
    url = 'https://datasets-server.huggingface.co/rows?' + query
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    # This host's HTTP stream can return short reads; collect
                    # chunks until EOF, then validate complete JSON.
                    chunks = []
                    while chunk := response.read(65536):
                        chunks.append(chunk)
                    payload = b''.join(chunks)
                # Check transport/JSON completeness before keeping the page.
                __import__('json').loads(payload)
                path.write_bytes(payload)
                break
            except Exception as error:
                if attempt == 2:
                    raise
                time.sleep(30 if isinstance(error, urllib.error.HTTPError) and error.code == 429 else 2)
    payload = load(path)
    # Never inspect/convert annotation percentages carried by the metadata API.
    return [{k: item['row'][k] for k in FIELDS} for item in payload['rows']], payload['num_rows_total'], url


def prepare(task):
    started = time.perf_counter()
    evidence = task / 'evidence'
    evidence.mkdir(exist_ok=True)
    assert not (evidence / 'freeze.json').exists(), 'Do not replace the committed population.'
    first, nrows, url = page(0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        other = list(pool.map(page, range(50, nrows, 50)))
    metadata = first + [row for rows, _, _ in other for row in rows]
    assert len(metadata) == nrows and len({r['datapoint_id'] for r in metadata}) == nrows
    high = [r for r in metadata if r['split'] == 'test' and str(r['proj_shape']) == '509' and r['label_type'] == 'high']
    assert len(high) == 975
    assert all(r['url'].endswith('/test/' + FILE_NAME) for r in high)
    old_meta_path = Path.home() / 'Data/onboard-acceptance/meta-cloudsen12plus/cloudsen12plus_high_trainval_metadata.csv'
    old = data.read_csv(old_meta_path)
    old_rois = {r['roi_id'] for r in old}
    excluded = [r for r in high if r['roi_id'] in old_rois]
    selected = sorted([r for r in high if r['roi_id'] not in old_rois], key=lambda r: r['datapoint_id'])
    assert selected and {r['roi_id'] for r in selected}.isdisjoint(old_rois)
    tree = load(DATA_ROOT / 'repository-test-tree.json')
    item = next(r for r in tree if r['path'] == 'test/' + FILE_NAME)
    assert item['size'] == 2679467254
    prior = load(PRIOR / 'freeze.json')
    heads = {k: v for k, v in load(PRIOR / 'models.json').items() if k.startswith('cloudsen_')}
    freeze = {'contract_revision': 1, 'dataset_revision': REVISION, 'url': URL,
              'cloudsen_file': str(DATA_ROOT / FILE_NAME), 'expected_file_bytes': item['size'],
              'scope': 'all eligible official test high-quality 509 patches; no performance-based selection',
              'n_patches': len(selected), 'n_rois': len({r['roi_id'] for r in selected}),
              'n_tiles': len(selected) * 49, 'scale': prior['scale'], 'heads': heads,
              'models_directory': prior['models_directory'], 'padding_side': prior['padding_side'],
              'runtime_order': ['single-1', 'ensemble-1', 'ensemble-2', 'single-2'],
              'bootstrap_draws': 2000, 'bootstrap_seed': 7,
              'selection': 'stable global top ceil(0.8*N)', 'new_fits': 0,
              'historical_labels': 128, 'primary_convention': 'A',
              'input_boundary': 'mapped four-band L1C image read through complete global A decision'}
    write_csv(evidence / 'manifest.csv', selected)
    data.dump(evidence / 'freeze.json', freeze)
    data.dump(evidence / 'data-access.json', {
        'authorization': 'User: create task and execute, after the explicit Pro test-ROI proposal on 17 October 2026.',
        'supersedes': 'Earlier no-CloudSEN-test-download boundary only for this frozen task; HYPSO remains reserved.',
        'dataset_revision': REVISION, 'metadata_url_example': url, 'metadata_pages': 1 + len(other),
        'metadata_rows': nrows, 'label_percentage_fields_used': False,
        'official_test_509_patches': len(high), 'excluded_overlapping_roi_patches': len(excluded),
        'excluded_roi_ids': sorted({r['roi_id'] for r in excluded}),
        'selected_patches': len(selected), 'selected_rois': freeze['n_rois'],
        'independent_of_retained_trainval_rois': True, 'reference_metadata': str(old_meta_path),
        'reference_roi_count': len(old_rois), 'image_download_bytes_expected': item['size'],
        'metadata_download_bytes': sum(p.stat().st_size for p in DATA_ROOT.glob('*.json')),
        'disk_free_bytes': shutil.disk_usage(DATA_ROOT).free,
        'new_directory': str(DATA_ROOT), 'script_outside_download_directory': True,
        'license': 'CC0, existing INVENTORY and retained official dataset card',
        'execution': execution(started), 'images_opened': False, 'labels_opened': False})
    print({k: freeze[k] for k in ('n_patches', 'n_rois', 'n_tiles', 'expected_file_bytes')}, flush=True)


def check_download(task):
    from common import fixed
    freeze, manifest = fixed(task)
    # Check the whole file's table, including any metadata-only excluded ROIs.
    high = []
    for path in sorted(DATA_ROOT.glob('test-metadata-50-*.json')):
        for item in load(path)['rows']:
            row = item['row']
            if row['split'] == 'test' and str(row['proj_shape']) == '509' and row['label_type'] == 'high':
                high.append({k: row[k] for k in FIELDS})
    result, _, _ = mlstac.check(freeze['cloudsen_file'], high)
    result['expected_file_bytes'] = freeze['expected_file_bytes']
    result['passes'] = bool(result['magic_ok'] and result['size_ok'] and result['header_length_matches_metadata'] and result['table_matches_metadata'] and result['file_bytes'] == freeze['expected_file_bytes'])
    result['raw_labels_read'] = False
    assert result['passes'], result
    data.dump(task / 'evidence/download.json', result)
    print(result)


if __name__ == '__main__':
    task = Path(sys.argv[1]).resolve()
    check_download(task) if '--check-download' in sys.argv else prepare(task)
