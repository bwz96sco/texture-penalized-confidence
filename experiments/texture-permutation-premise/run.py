"""One resumable full-existing-development target pass with committed source freeze."""
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from perm_features import extract, permutations
from perm_analysis import analyze
from common import dump, peak_mb, read_csv, setup, write_csv
from onboard import tiles
sys.path.append(str(Path(__file__).resolve().parents[1] / "cloudsen12-room"))
import mlstac


def main():
    setup()
    data, task = map(Path, sys.argv[1:3])
    evid, runs = task / "evidence", data / "runs/texture-permutation-premise"
    runs.mkdir(exist_ok=True)
    freeze_path = evid / "freeze.json"
    assert subprocess.run(["git", "ls-files", "--error-unmatch", str(freeze_path)], capture_output=True).returncode == 0
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(freeze_path)]).returncode == 0
    freeze = json.loads(freeze_path.read_text())
    assert subprocess.run(["git", "diff", "--quiet", freeze["code_commit"], "--", str(Path(__file__).parent),
                           "experiments/texture-full-diagnostic", "src/onboard", "experiments/cloudsen12-room/mlstac.py"]).returncode == 0
    manifest = json.loads((evid / "input-manifest.json").read_text())
    originals = read_csv("tasks/texture-full-diagnostic/evidence/tiles.csv")
    assert len(originals) == 30135
    for dataset, key in (("hypso", "hypso_units"), ("cloudsen", "cloudsen_units")):
        assert list(dict.fromkeys(r["unit"] for r in originals if r["dataset"] == dataset)) == manifest[key]
    meta = {r["datapoint_id"]: r for r in read_csv(data / "meta-cloudsen12plus/cloudsen12plus_high_trainval_metadata.csv")
            if r["split"] == "validation" and r["proj_shape"] == "509" and r["label_type"] == "high"}
    mlpath = Path(manifest["cloudsen_file"])
    checks, length, table = mlstac.check(mlpath, list(meta.values()))
    assert all(checks[k] for k in ("magic_ok", "size_ok", "header_length_matches_metadata", "table_matches_metadata"))
    assert set(meta) == set(manifest["cloudsen_units"])
    side = tuple(json.loads(Path("tasks/cloudsen12-room/evidence/download.json").read_text())["first_patch"]["padding_side_row_col_start"])
    state_path = runs / "progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "source_code_commit": freeze["code_commit"], "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "units": {}}
    assert state["source_code_commit"] == freeze["code_commit"]
    started, orders, features = time.perf_counter(), permutations(), {}
    for dataset, key in (("hypso", "hypso_units"), ("cloudsen", "cloudsen_units")):
        cache = runs / "units" / dataset
        cache.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(manifest[key]):
            path = cache / f"{name}.csv"
            if path.exists():
                records = read_csv(path)
            else:
                start = time.perf_counter()
                if dataset == "hypso":
                    x = np.load(data / "derived/hypso1" / f"{name}_toa.npy")
                    rgb = tiles.cut(x[..., 1:4], tiles.grid(*x.shape[:2]))
                else:
                    a, info = mlstac.read_patch(mlpath, length, *table[name], bands=[2, 3, 4])
                    assert info["count"] == 15 and tuple(info["shape"]) == (512, 512)
                    x = np.moveaxis(mlstac.crop(a, side), 0, -1).astype(np.float32) * np.float32(1e-4)
                    rgb = tiles.cut(x, tiles.grid(*x.shape[:2]))
                io_seconds = time.perf_counter() - start
                assert np.isfinite(rgb).all()
                records, timing = extract(rgb, freeze["legacy_freeze"], freeze["prototypes"], orders)
                write_csv(path, records)
                state["units"][f"{dataset}/{name}"] = {"n_tiles": len(records), "io_seconds": io_seconds, **timing}
                dump(state_path, state)
            assert len(records) == (140 if dataset == "hypso" else 49)
            features[(dataset, name)] = records
            if i % 25 == 0 or i == len(manifest[key]) - 1:
                print(json.dumps({"phase": "target_features", "dataset": dataset, "done": i + 1,
                                  "total": len(manifest[key]), "seconds": time.perf_counter() - started}), flush=True)
    rows = []
    for original in originals:
        feature = features[(original["dataset"], original["unit"])][int(original["tile"])]
        assert int(feature["tile"]) == int(original["tile"])
        for new, old in (("q_real", "lbp_q"), ("dc_real", "lbp_dc"), ("dn_real", "lbp_dn"), ("gradient", "gradient")):
            assert math.isclose(float(feature[new]), float(original[old]), abs_tol=1e-12, rel_tol=1e-12), (new, original["unit"])
        row = dict(original)
        row.update({key: float(value) for key, value in feature.items() if key not in row})
        rows.append(row)
    write_csv(evid / "tiles.csv", rows)
    print(json.dumps({"phase": "statistics", "tiles": len(rows)}), flush=True)
    statistics_start = time.perf_counter()
    analysis_timings = {}
    result = analyze(rows, freeze, analysis_timings)
    dump(evid / "results.json", result)
    now = dt.datetime.now(dt.timezone.utc)
    execution = {"status": "complete", "started_at": state["started_at"], "finished_at": now.isoformat(),
        "seconds_this_invocation": time.perf_counter() - started,
        "wall_seconds_since_start": (now - dt.datetime.fromisoformat(state["started_at"])).total_seconds(),
        "statistics_seconds": time.perf_counter() - statistics_start, "units": len(state["units"]),
        "analysis_timings": analysis_timings,
        "hypso_units": 28, "cloudsen_units": 535, "tiles": len(rows), "peak_rss_mb": peak_mb(),
        "source_code_commit": freeze["code_commit"],
        "target_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "timing_sums": {key: sum(r[key] for r in state["units"].values()) for key in
            ("io_seconds", "real_texture_seconds", "permutation_texture_seconds", "additional_scale_prototype_seconds")},
        "new_download_bytes": 0, "target_main_or_cnn_inferences": 0, "threads": 8,
        "tile_csv_bytes": (evid / "tiles.csv").stat().st_size, "freeze_bytes": freeze_path.stat().st_size}
    dump(evid / "execution.json", execution)
    print(json.dumps({"decision": result["decision"], "execution": execution}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        main()
