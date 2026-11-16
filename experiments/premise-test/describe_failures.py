"""Post-hoc description of the inspectable captures the model fails almost whole.

Usage: python -I describe_failures.py HYPSO1_DOWNLOAD HYPSO1_DERIVED MODELS EVIDENCE_DIR
Writes EVIDENCE_DIR/failure-captures.json:
- 178-niger: per-band share of pixels with any of the band's channels at the clip level
  (L1A DN, read with `hypso`), and its median band reflectance;
- 111-bangladesh against 25-bangladesh (same area, one day earlier): median band reflectance;
- 266-kotzbue: tile accuracy with its reflectance rescaled from hypso's solar zenith (at a
  misplaced centre, georef-sza.csv) to the zenith at Kotzebue Sound at the capture time.
Only inspectable captures; sets no threshold and changes nothing in the frozen evaluation.
"""
import contextlib, csv, datetime, io, json, math, sys

import numpy as np

download, hypso_dir, models, evid = sys.argv[1:5]
SAT_DN, SHAPE, BANDS = 9 * 4095, (956, 684), ("B1", "B2", "B3", "B4")
channels = json.load(open(f"{evid}/freeze.json"))["hypso"]["channels"]
out = {"note": "Post hoc, inspectable captures only; not decision-bearing."}


def medians(cap):
    x = np.load(f"{hypso_dir}/{cap}_toa.npy").reshape(-1, 4)
    return dict(zip(BANDS, np.median(x, axis=0).astype(float).tolist()))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # hypso imports matplotlib; never open GUI windows
    from hypso import Hypso1

    cap = "178-niger_2022-09-06T09-26-31Z"
    with contextlib.redirect_stdout(io.StringIO()):
        dn = np.asarray(Hypso1(f"{download}/{cap}/{cap.split('-', 1)[1]}-l1a.nc", verbose=False).l1a_cube)
    out[cap] = {"band_clip_share": {b: float((dn[..., channels[b]] >= SAT_DN).any(-1).mean()) for b in BANDS},
                "median_reflectance": medians(cap)}

    a, b = "111-bangladesh_2022-10-27T03-50-38Z", "25-bangladesh_2022-10-26T04-02-56Z"
    ma, mb = medians(a), medians(b)
    out["bangladesh"] = {a: ma, b: mb, "ratio": {k: ma[k] / mb[k] for k in BANDS}}

    from onboard import scoring, tiles
    from onboard.solar import sun_elevation
    cap = "266-kotzbue_2022-06-25T22-10-13Z"
    g = next(r for r in csv.DictReader(open(f"{evid}/georef-sza.csv")) if r["capture"] == cap)
    true_sza = 90 - sun_elevation(66.90, -162.60, datetime.datetime.fromisoformat(g["iso_time"]))
    scale = math.cos(math.radians(float(g["hypso_sza"]))) / math.cos(math.radians(true_sza))
    lab = np.fromfile(f"{download}/{cap}/{cap.split('-', 1)[1]}-slc-labels.bin", dtype=np.uint8).reshape(SHAPE)
    corners = tiles.grid(*SHAPE)
    _, lab_down = scoring.label_tiles(tiles.cut(lab, corners), 2, 0.5)
    main = scoring.load_member(f"{models}/member0.pt")
    acc = {}
    for name, s in (("as_run", 1.0), ("rescaled", scale)):
        x = tiles.cut(np.load(f"{hypso_dir}/{cap}_toa.npy") * np.float32(s), corners)
        pred_cf = (scoring.forward_tiles(main, x).argmax(-1) == 2).float().mean(-1).numpy()
        acc[name] = float(np.mean((pred_cf < 0.5) == lab_down))
    out[cap] = {"centre": [float(g["centre_lat"]), float(g["centre_lon"])], "hypso_sza": float(g["hypso_sza"]),
                "sza_at_kotzebue_sound": true_sza, "reflectance_scale": scale, "tile_accuracy": acc}

    json.dump(out, open(f"{evid}/failure-captures.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
