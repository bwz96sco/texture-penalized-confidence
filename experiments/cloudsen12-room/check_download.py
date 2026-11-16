"""Checks on the downloaded CloudSEN12+ validation file before the scored run (no model is run).

Usage: python -I check_download.py MLSTAC_FILE METADATA_CSV OUT_JSON URL REVISION
Checks the magic, header length, offset table and size against the metadata, then opens the first
patch: band count, data type, shape, label values and the padding side. Writes OUT_JSON.
"""
import csv, hashlib, json, sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import mlstac

path, meta_csv, out, url, rev = sys.argv[1:6]
rows = [r for r in csv.DictReader(open(meta_csv))
        if r["split"] == "validation" and r["proj_shape"] == "509" and r["label_type"] == "high"]
assert len(rows) == 535
res, length, table = mlstac.check(path, rows)
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
res.update({"url": url, "revision": rev, "sha256": h.hexdigest()})
first = min(table, key=lambda k: table[k][0])
a, info = mlstac.read_patch(path, length, *table[first], bands=list(range(1, 16)))
img, lab = a[:4], a[13]
side = mlstac.padding_side(img, lab)
res["first_patch"] = {"id": first, **info, "label_values": sorted(int(v) for v in np.unique(lab)),
                      "padding_side_row_col_start": side,
                      "padding_zero": mlstac.padding_is_zero(img, lab, side) if side else None,
                      "b01_b04_dn_range": [int(img.min()), int(img.max())]}
res["passes"] = bool(res["magic_ok"] and res["size_ok"] and res["header_length_matches_metadata"]
                     and res["table_matches_metadata"] and info["count"] == 15 and side is not None
                     and res["first_patch"]["padding_zero"] and set(res["first_patch"]["label_values"]) <= {0, 1, 2, 3})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
