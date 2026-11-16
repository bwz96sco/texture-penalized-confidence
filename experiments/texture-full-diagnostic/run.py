"""One fixed all-development pass; completed units can resume after interruption.

Usage: uv run --no-sync python -I run.py DATA_ROOT TASK_DIR
"""
import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "cloudsen12-room"))
import mlstac
from common import (distances, dump, load_spatial, peak_mb, read_csv, setup,
                    spatial_scores, texture, write_csv)
from analysis import analyze
from onboard import tiles


def main():
    setup()
    data, task = map(Path, sys.argv[1:3])
    evid, runs = task / "evidence", data / "runs/texture-full-diagnostic"
    fzpath = evid / "freeze.json"
    assert subprocess.run(["git", "ls-files", "--error-unmatch", str(fzpath)], capture_output=True).returncode == 0
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(fzpath)]).returncode == 0
    freeze = json.loads(fzpath.read_text())
    assert subprocess.run(["git", "diff", "--quiet", freeze.get("target_code_commit", freeze["code_commit"]), "--",
                           str(Path(__file__).parent), "src/onboard",
                           "experiments/cloudsen12-room/mlstac.py"]).returncode == 0
    model_path = Path(freeze["spatial_model"])
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == freeze["spatial_sha256"]
    model = load_spatial(model_path, freeze["normalization_mean"], freeze["normalization_std"])
    manifest = json.loads((evid / "input-manifest.json").read_text())
    hrows = [r for r in read_csv("tasks/premise-test/evidence/tiles-shift-v0.csv") if r["reserved"] == "0"]
    crows = read_csv("tasks/cloudsen12-room/evidence/room-tiles.csv")
    assert list(dict.fromkeys(r["capture"] for r in hrows)) == manifest["hypso_units"]
    assert list(dict.fromkeys(r["patch"] for r in crows)) == manifest["cloudsen_units"]
    reserved_names = {r["capture"] for r in read_csv("tasks/premise-test/evidence/tiles-shift-v0.csv") if r["reserved"] == "1"}
    assert not set(manifest["hypso_units"]) & reserved_names
    meta = {r["datapoint_id"]: r for r in read_csv(data / "meta-cloudsen12plus/cloudsen12plus_high_trainval_metadata.csv")
            if r["split"] == "validation" and r["proj_shape"] == "509" and r["label_type"] == "high"}
    dl = json.loads(Path("tasks/cloudsen12-room/evidence/download.json").read_text())
    mlpath = Path(manifest["cloudsen_file"])
    checks, length, table = mlstac.check(mlpath, list(meta.values()))
    assert all(checks[k] for k in ("magic_ok", "size_ok", "header_length_matches_metadata", "table_matches_metadata"))
    assert set(meta) == set(manifest["cloudsen_units"])
    side = tuple(dl["first_patch"]["padding_side_row_col_start"])
    state_path = runs / "target-progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "code_commit": freeze["code_commit"], "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "units": {}}
    assert state["code_commit"] == freeze["code_commit"]
    started = time.perf_counter()
    features = {}
    for dataset, units in (("hypso", manifest["hypso_units"]), ("cloudsen", manifest["cloudsen_units"])):
        cache = runs / "units" / dataset
        cache.mkdir(parents=True, exist_ok=True)
        for number, name in enumerate(units):
            path = cache / f"{name}.csv"
            if path.exists():
                rows = read_csv(path)
                assert len(rows) == (140 if dataset == "hypso" else 49)
            else:
                t = time.perf_counter()
                if dataset == "hypso":
                    assert name not in reserved_names
                    x = np.load(data / "derived/hypso1" / f"{name}_toa.npy")
                    rgb = tiles.cut(x[..., 1:4], tiles.grid(*x.shape[:2]))
                else:
                    a, info = mlstac.read_patch(mlpath, length, *table[name], bands=[2, 3, 4])
                    assert info["count"] == 15 and tuple(info["shape"]) == (512, 512)
                    x = np.moveaxis(mlstac.crop(a, side), 0, -1).astype(np.float32) * np.float32(1e-4)
                    rgb = tiles.cut(x, tiles.grid(*x.shape[:2]))
                io_seconds = time.perf_counter() - t
                assert np.isfinite(rgb).all()
                t = time.perf_counter()
                hist, grad = texture(rgb)
                dc, dn, q = distances(hist, freeze["templates"])
                texture_seconds = time.perf_counter() - t
                t = time.perf_counter()
                spatial, conf = spatial_scores(model, rgb)
                cnn_seconds = time.perf_counter() - t
                rows = [{"tile": i, "lbp_dc": float(dc[i]), "lbp_dn": float(dn[i]),
                         "lbp_q": float(q[i]), "gradient": float(grad[i]),
                         "spatial_cloud_frac": float(spatial[i]), "spatial_conf": float(conf[i])}
                        for i in range(len(rgb))]
                assert len(rows) == (140 if dataset == "hypso" else 49)
                write_csv(path, rows)
                state["units"][f"{dataset}/{name}"] = {"n_tiles": len(rows), "io_seconds": io_seconds,
                    "texture_seconds": texture_seconds, "cnn_seconds": cnn_seconds}
                dump(state_path, state)
            features[(dataset, name)] = rows
            if number % 10 == 0 or number == len(units) - 1:
                print(json.dumps({"phase": "target", "dataset": dataset, "done": number + 1,
                                  "total": len(units), "seconds_this_invocation": time.perf_counter() - started}), flush=True)
    records = []
    for dataset, originals, key in (("hypso", hrows, "capture"), ("cloudsen", crows, "patch")):
        for r in originals:
            name, tile = r[key], int(r["tile"])
            f = features[(dataset, name)][tile]
            assert int(f["tile"]) == tile
            base = {"dataset": dataset, "unit": name, "cluster": name if dataset == "hypso" else r["roi"], "tile": tile}
            base.update({k: float(r[k]) for k in ("pred_cloud_frac", "sat_frac", "conf", "ens")})
            base["label_cloud_frac"] = float(r["label_cloud_frac"] if dataset == "hypso" else r["thick_frac"])
            base["thin_frac"] = 0.0 if dataset == "hypso" else float(r["thin_frac"])
            base["sza"] = 0.0 if dataset == "hypso" else float(r["sza"])
            base.update({k: float(v) for k, v in f.items() if k != "tile"})
            records.append(base)
    assert len(records) == 30135
    write_csv(evid / "tiles.csv", records)
    t = time.perf_counter()
    result = analyze(records, freeze)
    dump(evid / "results.json", result)
    execution = {"status": "complete", "started_at": state["started_at"],
                 "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "seconds_this_invocation": time.perf_counter() - started,
                 "wall_seconds_since_first_target_start": (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(state["started_at"])).total_seconds(),
                 "summary_seconds": time.perf_counter() - t, "units": len(state["units"]),
                 "hypso_units": len(manifest["hypso_units"]), "cloudsen_units": len(manifest["cloudsen_units"]),
                 "tiles": len(records), "target_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "source_code_commit": freeze["code_commit"], "peak_rss_mb": peak_mb(),
                 "platform": platform.platform(), "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                 "threads": 8, "device": "cpu", "new_download_bytes": 0,
                 "spatial_model_bytes": model_path.stat().st_size, "tile_csv_bytes": (evid / "tiles.csv").stat().st_size,
                 "timing_sums": {k: sum(u[k] for u in state["units"].values())
                                  for k in ("io_seconds", "texture_seconds", "cnn_seconds")}}
    dump(evid / "execution.json", execution)
    print(json.dumps({"complete": True, "decision": result["decision"], "execution": execution}, indent=2), flush=True)


if __name__ == "__main__":
    main()
