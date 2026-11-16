"""Fit all source-only prototypes/maps, smoke actual invariance and freeze before target."""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from perm_features import (K, RADII, SEED, PRIMARY, MATCHED, extract, fit_mappings,
                           fit_prototypes, lbp_hist, multi_distances, permutations,
                           permute, source_probability)
from common import distances, dump, load_spatial, peak_mb, read_csv, setup, texture, write_csv
from onboard import gates, scoring, tiles


def measure(fn, n):
    fn(); fn()
    seconds = []
    for _ in range(3):
        start = time.perf_counter()
        fn()
        seconds.append(time.perf_counter() - start)
    return {"tiles": n, "seconds": seconds, "median_ms_per_tile": float(np.median(seconds) * 1000 / n)}


def main():
    setup()
    data, task = map(Path, sys.argv[1:3])
    evid = task / "evidence"
    evid.mkdir(exist_ok=True)
    code = Path(__file__).parent
    assert not subprocess.check_output(["git", "status", "--porcelain", "--", str(code)]).strip(), "commit source code first"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started, started_at = time.perf_counter(), dt.datetime.now(dt.timezone.utc).isoformat()
    old = Path("tasks/texture-full-diagnostic/evidence")
    legacy = json.loads((old / "freeze.json").read_text())
    manifest = json.loads((old / "input-manifest.json").read_text())
    support = json.loads(Path("tasks/texture-residual-mechanism/evidence/calibration.json").read_text())
    reference = read_csv("tasks/premise-test/evidence/tiles-source-ref.csv")
    histograms = {radius: [] for radius in RADII}
    labels, details, io_seconds, train_feature_seconds = [], [], 0.0, 0.0
    for name in sorted(manifest["source_split"]["train"]):
        start = time.perf_counter()
        x = np.load(data / "derived/sparcs" / f"{name}_toa.npy")
        raw = np.load(data / "derived/sparcs" / f"{name}_labels.npy")
        rgb = tiles.cut(np.nan_to_num(x[..., 1:4], nan=float(np.mean(legacy["normalization_mean"])),
                                     posinf=0, neginf=0), tiles.grid(*raw.shape))
        label = tiles.cut(raw == 5, tiles.grid(*raw.shape)).mean((1, 2)) >= .5
        io_seconds += time.perf_counter() - start
        start = time.perf_counter()
        for radius in RADII:
            histograms[radius].append(lbp_hist(rgb, radius))
        train_feature_seconds += time.perf_counter() - start
        labels.extend(label.tolist())
        details.append({"split": "train", "unit": name, "tiles": len(rgb), "cloud_tiles": int(label.sum())})
    histograms = {radius: np.concatenate(value) for radius, value in histograms.items()}
    labels = np.asarray(labels)
    assert len(labels) == 13950 and int(labels.sum()) == 2167
    start = time.perf_counter()
    prototypes, prototype_report = fit_prototypes(histograms, labels)
    prototype_seconds = time.perf_counter() - start
    del histograms
    source, dev_times, bench_four = [], {}, None
    orders = permutations()
    for name in sorted(manifest["source_split"]["dev"]):
        start = time.perf_counter()
        x = np.load(data / "derived/sparcs" / f"{name}_toa.npy")
        raw = np.load(data / "derived/sparcs" / f"{name}_labels.npy")
        corners = tiles.grid(*raw.shape)
        x_tiles = tiles.cut(x, corners)
        rgb = np.nan_to_num(x_tiles[..., 1:4], nan=float(np.mean(legacy["normalization_mean"])), posinf=0, neginf=0)
        io_seconds += time.perf_counter() - start
        records, times = extract(rgb, legacy, prototypes, orders)
        original = [r for r in reference if r["subscene"] == name]
        fraction = tiles.cut(raw == 5, corners).mean((1, 2))
        assert len(records) == len(original) == 225
        assert np.array_equal(fraction, [float(r["label_cloud_frac"]) for r in original])
        for r, feature in zip(original, records):
            assert int(r["tile"]) == feature["tile"]
            source.append({**r, **{key: value for key, value in feature.items() if key != "tile"}})
        for key, value in times.items():
            dev_times[key] = dev_times.get(key, 0.0) + value
        details.append({"split": "dev", "unit": name, "tiles": len(records), "cloud_tiles": int((fraction >= .5).sum())})
        if bench_four is None:
            bench_four = x_tiles[:32].copy()
    assert len(source) == 3600
    start = time.perf_counter()
    mappings = fit_mappings(source)
    mapping_seconds = time.perf_counter() - start
    members = [scoring.load_member(str(data / "runs/premise-test/models" / f"member{k}.pt"), 64) for k in range(5)]
    sample = bench_four[:8]
    before = gates.tile_scores(members[0], members[1:], sample)
    invariance = {key: 0.0 for key in ("pred_cloud_fraction", "conf", "ens")}
    for order in orders:
        after = gates.tile_scores(members[0], members[1:], permute(sample, order))
        for key in invariance:
            delta = float(np.abs(before[key] - after[key]).max())
            assert delta <= 1e-6, (key, delta)
            invariance[key] = max(invariance[key], delta)
    rgb = bench_four[..., 1:4]

    def primary_pipeline():
        q = distances(lbp_hist(rgb, 3), legacy["templates"])[2]
        null = np.mean([distances(lbp_hist(permute(rgb, order), 3), legacy["templates"])[2] for order in orders], axis=0)
        return source_probability([{"a": float(a)} for a in q - null], mappings[PRIMARY])

    def q_pipeline():
        q = distances(lbp_hist(rgb, 3), legacy["templates"])[2]
        return source_probability([{"q_real": float(qi)} for qi in q], mappings[MATCHED])

    def multi_pipeline():
        q = np.mean([multi_distances(lbp_hist(rgb, r), prototypes[str(r)]) for r in RADII], axis=0)
        return source_probability([{"q_multiscale": float(qi)} for qi in q], mappings["multi-scale-template/v1"])

    cnn = load_spatial(legacy["spatial_model"], legacy["normalization_mean"], legacy["normalization_std"])
    tx = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float()
    flat = torch.from_numpy(bench_four.reshape(-1, 4))
    with torch.no_grad():
        benchmarks = {"primary_feature_and_map": measure(primary_pipeline, 32),
            "matched_q_feature_and_map": measure(q_pipeline, 32),
            "multi_scale_feature_and_map": measure(multi_pipeline, 32),
            "real_texture_pair": measure(lambda: texture(rgb), 32),
            "main_mlp_forward": measure(lambda: members[0](flat), 32),
            "five_mlp_forward": measure(lambda: [m(flat) for m in members], 32),
            "spatial_cnn_forward": measure(lambda: cnn(tx), 32)}
    freeze = {"contract_revision": 1, "code_commit": revision,
        "legacy_freeze": legacy, "prototypes": prototypes, "mappings": mappings,
        "support_cloud_95": support["cloud_radius_95"], "support_clear_95": support["clear_radius_95"],
        "permutation_seed": SEED, "permutation_count": K,
        "permutation_policy": "four fixed joint pixel-index permutations shared by every tile",
        "radii": list(RADII), "mapping_recipe": {"C": 1, "solver": "lbfgs", "tol": 1e-8,
            "max_iter": 200, "source_population": "all source-dev main-cloud calls", "standardization": "population mean/SD"},
        "primary": PRIMARY, "penalty_coefficient": 1, "coverage": .8,
        "bootstrap_draws": 2000, "bootstrap_seed": 7, "fit_population": "SPARCS 62 train / 16 dev only"}
    smoke = json.loads((evid / "smoke.json").read_text())
    smoke["actual_source_ensemble_invariance"] = {"tiles": 8, "permutations": K,
        "max_absolute_difference": invariance, "tolerance": 1e-6}
    smoke["source_probability_export_roundtrip"] = "all seven mappings agree with sklearn predict_proba to 1e-12"
    elapsed = time.perf_counter() - started
    assert elapsed < 600, ("source allowance exceeded", elapsed)
    report = {"status": "complete", "started_at": started_at, "source_units": details,
        "train_tiles": len(labels), "train_cloud_tiles": int(labels.sum()), "dev_tiles": len(source),
        "dev_cloud_calls": 1034, "dev_cloud_call_errors": 341, "source_io_seconds": io_seconds,
        "train_histogram_seconds": train_feature_seconds, "prototype_fit_seconds": prototype_seconds,
        "dev_feature_times": dev_times, "mapping_fit_seconds": mapping_seconds,
        "total_seconds": elapsed, "peak_rss_mb": peak_mb(), "prototype_report": prototype_report,
        "benchmarks": benchmarks, "code_commit": revision, "threads": 8,
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "target_images_read": 0, "new_cnn_training": False,
        "source_model_inference_purpose": "8-tile invariance and 32-tile warmed cost benchmarks only"}
    write_csv(evid / "source-features.csv", source)
    dump(evid / "freeze.json", freeze)
    dump(evid / "input-manifest.json", manifest)
    dump(evid / "smoke.json", smoke)
    dump(evid / "source.json", report)
    print(json.dumps({"source_complete": True, "seconds": elapsed, "invariance": invariance,
                      "mapping_coefficients": {name: fit["coef"] for name, fit in mappings.items()}}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        main()
