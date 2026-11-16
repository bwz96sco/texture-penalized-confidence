"""Full source fit, source-only checkpoint selection and cost smoke measurements.

Usage: uv run --no-sync python -I prepare.py DATA_ROOT TASK_DIR
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import (BATCH, EPOCHS, SEED, Spatial, dump, peak_mb, read_csv,
                    setup, spatial_scores, texture)
from onboard import scoring, tiles


def all_tiles(x, labels):
    """Cover all real source pixels exactly once, with ignored padded targets."""
    h, w = labels.shape
    hp, wp = (-h) % 64, (-w) % 64
    xx = np.pad(x, ((0, hp), (0, wp), (0, 0)), mode="reflect")
    yy = np.pad(labels, ((0, hp), (0, wp)), constant_values=255)
    corners = tiles.grid(*yy.shape)
    return tiles.cut(xx, corners), tiles.cut(yy, corners)


@torch.no_grad()
def dev_loss(model, x, y, weights):
    model.eval()
    numerator = denominator = 0.0
    for k in range(0, len(x), BATCH):
        yy = y[k:k + BATCH]
        valid = yy != 255
        if not valid.any():
            continue
        logits = model(x[k:k + BATCH])
        numerator += F.cross_entropy(logits, yy, weight=weights, ignore_index=255,
                                     reduction="sum").item()
        denominator += weights[yy[valid]].sum().item()
    return numerator / denominator


def measure(fn, ntiles):
    with torch.no_grad():
        fn()
        fn()
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - t)
    return {"tiles_per_batch": ntiles, "seconds": runs,
            "median_ms_per_tile": float(np.median(runs) * 1000 / ntiles)}


def main():
    setup()
    data, task = map(Path, sys.argv[1:3])
    evid = task / "evidence"
    evid.mkdir(exist_ok=True)
    runs = data / "runs" / "texture-full-diagnostic"
    runs.mkdir(exist_ok=True)
    code = Path(__file__).parent
    assert not subprocess.check_output(["git", "status", "--porcelain", "--", str(code)]).strip(), "commit code first"
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.time()
    pf = json.loads(Path("tasks/premise-test/evidence/freeze.json").read_text())
    members = []
    for k, digest in pf["model"]["member_sha256"].items():
        p = data / "runs/premise-test/models" / f"member{k}.pt"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        members.append(scoring.load_member(str(p), 64))
    mean = members[0].mean.detach().numpy()[1:4].copy()
    std = members[0].std.detach().numpy()[1:4].copy()
    source = data / "derived/sparcs"
    split = pf["split"]
    ref = read_csv("tasks/premise-test/evidence/tiles-source-ref.csv")
    xx, yy, sums = {}, {}, {"cloud": np.zeros(36), "clear": np.zeros(36)}
    ntemplates = {"cloud": 0, "clear": 0}
    grad_cloud, eval_rgb, eval_cloud = [], [], []
    counts = np.zeros(2, dtype=np.int64)
    details = []
    bench_four = None
    for group in ("train", "dev"):
        xs, ys = [], []
        for name in sorted(split[group]):
            x4 = np.load(source / f"{name}_toa.npy")
            raw = np.load(source / f"{name}_labels.npy")
            # Same binary supervision as the texture templates and tile endpoint;
            # retaining snow/shadow as non-cloud avoids an asymmetric label omission.
            lab = (raw == 5).astype(np.uint8)
            finite = np.isfinite(x4).all(-1)
            lab[~finite] = 255
            x = np.nan_to_num(x4[..., 1:4], nan=float(mean.mean()), posinf=0, neginf=0)
            corners = tiles.grid(*lab.shape)
            rgb = tiles.cut(x, corners)
            cloud = tiles.cut(raw == 5, corners).mean((1, 2)) >= 0.5
            h, g = texture(rgb)
            if group == "train":
                for key, mask in (("cloud", cloud), ("clear", ~cloud)):
                    sums[key] += h[mask].sum(0)
                    ntemplates[key] += int(mask.sum())
                counts += np.bincount(lab[lab != 255], minlength=2)
            else:
                grad_cloud.extend(g[cloud].tolist())
                eval_rgb.append(rgb)
                records = [r for r in ref if r["subscene"] == name]
                assert len(records) == len(rgb)
                assert np.array_equal(cloud, [float(r["label_cloud_frac"]) >= 0.5 for r in records])
                eval_cloud.extend(cloud.tolist())
                if bench_four is None:
                    bench_four = tiles.cut(x4, corners)[:32].copy()
            xt, yt = all_tiles(x, lab)
            assert int((yt != 255).sum()) == int((lab != 255).sum())
            xs.append(xt)
            ys.append(yt)
            details.append({"split": group, "name": name, "full_training_tiles": len(xt),
                            "eligible_eval_tiles": len(rgb), "valid_pixels": int((lab != 255).sum())})
        xx[group] = torch.from_numpy(np.ascontiguousarray(np.concatenate(xs).transpose(0, 3, 1, 2))).float()
        yy[group] = torch.from_numpy(np.concatenate(ys).astype(np.int64))
        del xs, ys
        print(json.dumps({"phase": "source_prepared", "split": group, "tiles": len(xx[group]),
                          "valid_pixels": int((yy[group] != 255).sum())}), flush=True)
    assert all(ntemplates.values()) and grad_cloud
    templates = {k: (sums[k] / ntemplates[k]).tolist() for k in sums}
    gradient_threshold = float(np.quantile(grad_cloud, 0.95))
    prepared_seconds = time.time() - started
    weights = torch.from_numpy((counts.sum() / (2 * counts)).astype(np.float32))
    model = Spatial(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(SEED)
    history, best, selected = [], float("inf"), None
    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t = time.time()
        model.train()
        order = torch.randperm(len(xx["train"]), generator=generator)
        seen = 0
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            y = yy["train"][idx]
            valid = y != 255
            seen += int(valid.sum())
            if not valid.any():
                continue
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xx["train"][idx]), y, weight=weights, ignore_index=255)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        assert seen == int(counts.sum())
        entry = {"epoch": epoch, "training_valid_pixels_seen": seen, "seconds": time.time() - t}
        if epoch % 5 == 0:
            entry["source_dev_weighted_ce"] = dev_loss(model, xx["dev"], yy["dev"], weights)
            if entry["source_dev_weighted_ce"] < best:
                best, selected = entry["source_dev_weighted_ce"], epoch
                torch.save(model.state_dict(), runs / "spatial.pt")
        history.append(entry)
        dump(runs / "training-progress.json", history)
        print(json.dumps(entry), flush=True)
    train_seconds = time.time() - train_start
    model.load_state_dict(torch.load(runs / "spatial.pt", weights_only=True))
    model.eval()
    evaluation_rgb = np.concatenate(eval_rgb)
    sc, sf = spatial_scores(model, evaluation_rgb)
    source_accuracy = float(((sc >= 0.5) == np.asarray(eval_cloud)).mean())
    assert len(sc) == len(ref) == 3600
    del xx, yy, evaluation_rgb, eval_rgb

    rgb = bench_four[..., 1:4]
    tx = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float()
    flat = torch.from_numpy(bench_four.reshape(-1, 4))
    benchmarks = {"texture_both_descriptors": measure(lambda: texture(rgb), len(rgb)),
                  "spatial_cnn": measure(lambda: model(tx), len(rgb)),
                  "main_mlp": measure(lambda: members[0](flat), len(rgb)),
                  "five_mlp_forward": measure(lambda: [m(flat) for m in members], len(rgb))}
    source_report = {"source_split": split, "full_coverage": details, "class_counts": counts.tolist(),
                     "class_order": ["non-cloud", "cloud"], "label_mapping": "SPARCS code 5 cloud, all other codes non-cloud",
                     "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
                     "template_tiles": ntemplates, "gradient_true_cloud_tiles": len(grad_cloud),
                     "gradient_95pct": gradient_threshold, "history": history, "selected_epoch": selected,
                     "source_dev_tile_accuracy": source_accuracy, "spatial_control_adequate": source_accuracy >= 0.85,
                     "source_dev_tiles": len(sc), "parameters": sum(p.numel() for p in model.parameters()),
                     "source_prepare_seconds": prepared_seconds, "training_seconds": train_seconds,
                     "total_seconds": time.time() - started, "peak_rss_mb": peak_mb(),
                     "benchmarks": benchmarks, "torch_version": torch.__version__, "threads": 8,
                     "device": "cpu", "code_commit": rev}
    dump(evid / "source.json", source_report)
    hypso_rows = [r for r in read_csv("tasks/premise-test/evidence/tiles-shift-v0.csv") if r["reserved"] == "0"]
    cloud_rows = read_csv("tasks/cloudsen12-room/evidence/room-tiles.csv")
    manifest = {"hypso_units": list(dict.fromkeys(r["capture"] for r in hypso_rows)),
                "cloudsen_units": list(dict.fromkeys(r["patch"] for r in cloud_rows)),
                "hypso_tiles": len(hypso_rows), "cloudsen_tiles": len(cloud_rows),
                "source_split": split, "data_root": str(data),
                "cloudsen_file": str(data / "cloudsen12plus-val509/validation_509_high.mlstac")}
    assert len(manifest["hypso_units"]) == 28 and len(manifest["cloudsen_units"]) == 535
    dump(evid / "input-manifest.json", manifest)
    revision = json.loads(subprocess.check_output(["rctl", "--format", "json", "status", str(task)], text=True))["data"]["contract_revision"]
    freeze = {"contract_revision": revision, "code_commit": rev, "templates": templates,
              "gradient_threshold": gradient_threshold, "normalization_mean": mean.tolist(),
              "normalization_std": std.tolist(), "spatial_model": str(runs / "spatial.pt"),
              "spatial_sha256": hashlib.sha256((runs / "spatial.pt").read_bytes()).hexdigest(),
              "selected_epoch": selected, "source_dev_accuracy": source_accuracy,
              "spatial_control_adequate": source_accuracy >= 0.85,
              "source_ensemble_hashes": pf["model"]["member_sha256"],
              "fit_population": "SPARCS 62 train / 16 dev only", "coverage": 0.8,
              "bootstrap_draws": 2000, "bootstrap_seed": 7,
              "primary_texture_arm": "lbp-continuous/v1"}
    dump(evid / "freeze.json", freeze)
    lines = ["# Source and synthetic smoke checks", "", "Synthetic checks passed before this source run.",
             "Full 62/16 source coverage is recorded in source.json; no target images read.",
             f"Selected epoch: {selected}; source-dev tile accuracy {source_accuracy:.6f}.",
             f"Spatial control adequate at the fixed 0.85 floor: {source_accuracy >= 0.85}.",
             f"Source fitting/selection total seconds: {source_report['total_seconds']:.2f}.",
             "The target settings now remain fixed; no target fitting or architecture search.", ""]
    (evid / "smoke.md").write_text("\n".join(lines))
    print(json.dumps({"source_done": True, "selected_epoch": selected,
                      "accuracy": source_accuracy, "seconds": source_report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
