"""Read a CloudSEN12+ v1.0 `.mlstac` file without the mlstac package.

Layout (mlstac 0.1.7 reader): bytes 0-1 magic b"#y"; bytes 2-9 uint64 little-endian header
length L; then a JSON header {datapoint_id: [begin, length]}; then the patches back to back, each a
raster rasterio opens from its byte range at 10 + L + begin.
"""
import json, os, struct

import numpy as np

PAD = 3          # 509 -> 512
SIZE = 509


def header(path):
    with open(path, "rb") as f:
        magic = f.read(2)
        (length,) = struct.unpack("<Q", f.read(8))
        table = json.loads(f.read(length))
    return magic, length, table


def check(path, meta_rows):
    """Header, offset table and size against the metadata rows of this file."""
    magic, length, table = header(path)
    size = os.path.getsize(path)
    end = max(b + n for b, n in table.values())
    out = {"magic_ok": magic == b"#y", "header_length": length, "n_patches": len(table), "file_bytes": size,
           "size_ok": size == 10 + length + end,
           "header_length_matches_metadata": all(int(r["metadata_length"]) == length for r in meta_rows),
           "table_matches_metadata": {r["datapoint_id"]: [int(r["begin"]), int(r["length"])] for r in meta_rows} ==
                                     {k: [int(v[0]), int(v[1])] for k, v in table.items()}}
    return out, length, table


def read_patch(path, length, begin, nbytes, bands):
    import rasterio
    with rasterio.open(f"/vsisubfile/{10 + length + begin}_{nbytes},{path}") as src:
        return src.read(bands), {"count": src.count, "dtypes": list(src.dtypes), "shape": [src.height, src.width],
                                 "driver": src.driver}


def crop(a, side):
    """Remove the zero padding: side is (row_start, col_start) of the 509 x 509 window."""
    r, c = side
    return a[..., r:r + SIZE, c:c + SIZE]


def padding_side(img, lab):
    """Find which rows/cols carry the padding: returns (row_start, col_start) or None if ambiguous."""
    nz_rows = np.flatnonzero((img != 0).any(axis=(0, 2)) | (lab != 0).any(axis=1))
    nz_cols = np.flatnonzero((img != 0).any(axis=(0, 1)) | (lab != 0).any(axis=0))
    h, w = lab.shape
    r = 0 if nz_rows.max() < SIZE else (PAD if nz_rows.min() >= PAD else None)
    c = 0 if nz_cols.max() < SIZE else (PAD if nz_cols.min() >= PAD else None)
    if h != SIZE + PAD or w != SIZE + PAD or r is None or c is None:
        return None
    return (r, c)


def padding_is_zero(img, lab, side):
    m = np.ones(lab.shape, bool)
    r, c = side
    m[r:r + SIZE, c:c + SIZE] = False
    return bool((img[:, m] == 0).all() and (lab[m] == 0).all())
