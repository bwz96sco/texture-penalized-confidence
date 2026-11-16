"""HYPSO-1 preprocessing (label-free): calibration with `hypso`, mapping to B1-B4, saturation.

Usage: python -I prep_hypso1.py DOWNLOAD_DIR OUT_DIR EVIDENCE_CSV [WORKERS]
Writes OUT_DIR/<capture>_toa.npy (H, W, 4 float32) and OUT_DIR/<capture>_sat.npy (pixels
with any channel at or above SAT_DN). Label files are not read.
"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from onboard import hypso1

# Saturation: 12-bit samples summed over HYPSO-1's 9-row spectral binning, 9 x 4095. Many
# pixels sit exactly at this value in L1A DN; a few isolated values exceed it.
SAT_DN = 9 * 4095

download, out, evidence = sys.argv[1], sys.argv[2], sys.argv[3]
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(out, exist_ok=True)


def one(nc_path):
    capture = os.path.basename(os.path.dirname(nc_path))
    r = hypso1.toa_reflectance(nc_path)
    ch = hypso1.band_channels(r["wavelengths"])
    x = hypso1.to_landsat_bands(r["toa"], ch)
    np.save(f"{out}/{capture}_toa.npy", x)
    sat = (r["dn"] >= SAT_DN).any(axis=-1)
    np.save(f"{out}/{capture}_sat.npy", sat)
    finite = np.isfinite(x)
    return {"capture": capture, "iso_time": r["iso_time"], "sza_mean": r["sza_mean"],
            "in_range": float(((x >= 0) & (x <= 1.5) & finite).mean()), "zero_share": float((x == 0).mean()),
            "dn_max": float(r["dn"].max()), "saturated_share": float(sat.mean()), "channels": ch,
            "shape": list(x.shape)}


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{download}/*/*-l1a.nc"))
    assert len(paths) == 38, len(paths)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, paths))
    channels = {json.dumps(r["channels"]) for r in rows}
    json.dump(rows[0]["channels"], open(f"{out}/channels.json", "w"))
    with open(evidence, "w") as f:
        f.write("capture,iso_time,sza_mean,in_range_0_1p5,zero_share,dn_max,saturated_share\n")
        for r in rows:
            f.write(f"{r['capture']},{r['iso_time']},{r['sza_mean']:.3f},{r['in_range']:.6f},{r['zero_share']:.6f},{r['dn_max']:.0f},{r['saturated_share']:.6f}\n")
    allin = np.mean([r["in_range"] for r in rows])
    print(f"captures {len(rows)}; mean share in [0, 1.5] {allin:.6f}; min {min(r['in_range'] for r in rows):.6f}; "
          f"saturation DN {SAT_DN}; channel sets identical: {len(channels) == 1}; channels {rows[0]['channels']}")
