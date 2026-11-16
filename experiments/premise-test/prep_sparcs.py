"""SPARCS preprocessing: TOA reflectance for B1-B4, merged labels, launch-check statistics.

Usage: python -I prep_sparcs.py SPARCS_DIR SUN_CSV OUT_DIR SUN_EVIDENCE_CSV
SUN_CSV comes from fetch_sparcs_sun.py. The one scene without Collection 2 metadata
takes the fallback below. Writes OUT_DIR/<subscene>_toa.npy (H, W, 4 float32, NaN where
any band is fill) and OUT_DIR/<subscene>_labels.npy (raw SPARCS codes).
"""
import csv, os, sys

import numpy as np

from onboard import sparcs

# LC81130672013142LGN01 is not in Collections 1 or 2. Its sun elevation is computed at
# the WRS-2 113/067 scene centre for 2013-05-22 at that path/row's scene-centre time on
# six later 2013 dates (NOAA equations; validated to 0.02 deg against 77 scenes'
# metadata). The six estimates span 49.504-49.518 deg; the median is used.
FALLBACK = {"LC81130672013142LGN01": 49.516}

sparcs_dir, sun_csv, out_dir, evidence_csv = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
sun = {}
for r in csv.DictReader(open(sun_csv)):
    if r["stac_id"] != "missing":
        sun[r["subscene"]] = (float(r["sun_elevation"]), f"stac:{r['stac_id']}")
    else:
        sun[r["subscene"]] = (FALLBACK[r["scene"]], "computed: NOAA, same path/row scene-centre time")

in_range = total = fill = 0
with open(evidence_csv, "w") as ev:
    ev.write("subscene,sun_elevation,source,fill_share\n")
    for name in sorted(sun):
        dn, labels = sparcs.load(sparcs_dir, name)
        rho = sparcs.toa_reflectance(dn, sun[name][0])
        bad = (dn == 0).any(axis=-1)
        rho[bad] = np.nan
        np.save(f"{out_dir}/{name}_toa.npy", rho)
        np.save(f"{out_dir}/{name}_labels.npy", labels)
        v = rho[~bad]
        in_range += int(((v >= 0) & (v <= 1.5)).sum()); total += v.size; fill += int(bad.sum())
        ev.write(f"{name},{sun[name][0]:.6f},{sun[name][1]},{bad.mean():.6f}\n")
print(f"subscenes {len(sun)}; values in [0, 1.5]: {in_range / total:.6f}; fill pixels {fill}")
