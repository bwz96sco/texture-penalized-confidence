"""Read-only check: the sample reconstructs from metadata, is independent, and its bytes are in place."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rule = load('untouched_select', Path(__file__).parent / 'select.py')
transfer = load('untouched_download', Path(__file__).parent / 'download.py')

task = Path(sys.argv[1]).resolve()
evidence = task / 'evidence'
selection, manifest = rule.select()  # validation pages are cached locally; no network
retained = json.loads((evidence / 'selection.json').read_text())
with open(evidence / 'manifest.csv', newline='') as f:
    rows = list(csv.DictReader(f))
assert selection == retained, 'selection does not reconstruct'
assert rows == manifest, 'manifest does not reconstruct'
assert list(rows[0]) == list(rule.FIELDS) and not any('percentage' in k for k in rows[0])
counts = retained['counts']
assert counts['sampled_rois'] == 330 and counts['manifest_patches'] == 1650 and counts['manifest_tiles'] == 80850
assert retained['nearest_used_km_sampled']['min'] > rule.RADIUS_KM
assert set(retained['sampled_rois']) | set(retained['reserve_rois']) == set(retained['eligible_rois'])

record = json.loads((evidence / 'download.json').read_text())
target = Path(record['local_path'])
assert target.stat().st_size == record['tree_size'] == 23622014997
with open(rule.TRAINVAL, newline='') as f:
    meta = [r for r in map(rule.project, csv.DictReader(f)) if r['url'].endswith('/' + rule.FILE)]
checks, length, table = transfer.mlstac().check(target, meta)
assert all(checks[k] for k in ('magic_ok', 'size_ok', 'header_length_matches_metadata', 'table_matches_metadata'))
covered = sorted((s['start'], s['end']) for s in record['spans_detail'])
for r in rows:
    a = 10 + length + int(r['begin'])
    b = a + int(r['length']) - 1
    assert any(s <= a and b <= e for s, e in covered), r['datapoint_id']
assert record['sparse_file_disk_bytes'] >= record['patch_bytes']
import rasterio  # noqa: E402
for r in rows:
    begin, nbytes = table[r['datapoint_id']]
    with rasterio.open(f'/vsisubfile/{10 + length + begin}_{nbytes},{target}') as src:  # header only
        assert src.count == 15 and (src.height, src.width) == (512, 512)
assert not (record['pixel_values_read'] or record['label_band_read'] or record['model_run'])
print('Untouched S2 sample reconstructs and its bytes are in place:', counts)
