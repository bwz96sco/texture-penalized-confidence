"""Post-freeze data-validity check: hypso's direct georeferencing and solar zenith per capture.

Usage: python -I check_georef.py HYPSO1_DOWNLOAD OUT_CSV
Label-free, all 38 captures: hypso's capture-centre latitude/longitude and solar zenith,
against the NOAA equations at that centre and time. The effect of the one misplaced capture
is in describe_failures.py. Sets no threshold and changes nothing in the frozen evaluation.
"""
import contextlib, csv, datetime, glob, io, sys
from concurrent.futures import ProcessPoolExecutor

download, out = sys.argv[1:3]


def one(path):
    import matplotlib
    matplotlib.use("Agg")  # hypso imports matplotlib; never open GUI windows
    from hypso import Hypso1
    from onboard.solar import sun_elevation
    with contextlib.redirect_stdout(io.StringIO()):
        h = Hypso1(path, verbose=False)
        h.run_direct_georeferencing()
    lat, lon, sza = h.latitudes_direct, h.longitudes_direct, h.solar_zenith_angles_direct
    r, c = lat.shape[0] // 2, lat.shape[1] // 2
    when = datetime.datetime.fromisoformat(str(h.iso_time))
    return {"capture": path.split("/")[-2], "iso_time": str(h.iso_time), "centre_lat": float(lat[r, c]),
            "centre_lon": float(lon[r, c]), "hypso_sza": float(sza[r, c]),
            "noaa_sza": 90 - sun_elevation(float(lat[r, c]), float(lon[r, c]), when)}


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(one, sorted(glob.glob(f"{download}/*/*-l1a.nc"))))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"max |hypso - NOAA| at the georeferenced centre: {max(abs(r['hypso_sza'] - r['noaa_sza']) for r in rows):.2f} deg")

