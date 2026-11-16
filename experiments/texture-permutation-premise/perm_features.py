"""Frozen same-tile contrast and source-only continuous/strong-texture controls."""
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texture-full-diagnostic"))
from common import LBP_BINS, distances, texture

SEED, K, RADII = 20261015, 4, (1, 3, 9)
PRIMARY, MATCHED, NESTED = "permutation-contrast/v1", "calibrated-q/v1", "q-gradient-contrast/v1"
FEATURES = {PRIMARY: ("a",), MATCHED: ("q_real",),
            "continuous-gradient/v1": ("gradient",),
            "multi-template/v1": ("q_multi3",),
            "multi-scale-template/v1": ("q_multiscale",),
            "q-gradient/v1": ("q_real", "gradient"),
            NESTED: ("q_real", "gradient", "a")}


def permutations():
    rng = np.random.default_rng(SEED)
    return [rng.permutation(4096) for _ in range(K)]


def permute(rgb, order):
    return rgb.reshape(len(rgb), 4096, -1)[:, order].reshape(rgb.shape)


def lbp_hist(rgb, radius):
    gray = np.asarray(rgb, dtype=np.float64).mean(-1)
    size = 64 - 2 * radius
    center = gray[:, radius:-radius, radius:-radius]
    code = np.zeros(center.shape, dtype=np.uint8)
    for k in range(8):
        dy = round(radius * math.sin(2 * math.pi * k / 8), 12)
        dx = round(radius * math.cos(2 * math.pi * k / 8), 12)
        y, x = math.floor(radius + dy), math.floor(radius + dx)
        fy, fx = radius + dy - y, radius + dx - x
        near = (1 - fy) * (1 - fx) * gray[:, y:y + size, x:x + size]
        if fx:
            near = near + (1 - fy) * fx * gray[:, y:y + size, x + 1:x + size + 1]
        if fy:
            near = near + fy * (1 - fx) * gray[:, y + 1:y + size + 1, x:x + size]
        if fx and fy:
            near = near + fy * fx * gray[:, y + 1:y + size + 1, x + 1:x + size + 1]
        code |= (near >= center).astype(np.uint8) << k
    offsets = 36 * np.arange(len(rgb), dtype=np.int64)[:, None, None]
    return np.bincount((LBP_BINS[code] + offsets).ravel(), minlength=36 * len(rgb)).reshape(len(rgb), 36) / size ** 2


def multi_distances(hist, prototypes):
    values = []
    for key in ("cloud", "clear"):
        centers = np.asarray(prototypes[key])
        distance = 0.5 * (((hist[:, None] - centers[None]) ** 2) /
                          (hist[:, None] + centers[None] + 1e-12)).sum(-1)
        values.append(distance.min(1))
    dc, dn = values
    return (dc - dn) / (dc + dn + 1e-12)


def fit_prototypes(histograms, cloud):
    models, report = {}, {}
    with threadpool_limits(limits=8):
        for radius in RADII:
            models[str(radius)], report[str(radius)] = {}, {}
            for name, mask in (("cloud", cloud), ("clear", ~cloud)):
                fit = KMeans(n_clusters=4, init="k-means++", n_init=1, random_state=SEED,
                             max_iter=100, algorithm="lloyd").fit(histograms[radius][mask])
                models[str(radius)][name] = fit.cluster_centers_.tolist()
                report[str(radius)][name] = {"n_tiles": int(mask.sum()), "iterations": int(fit.n_iter_),
                    "inertia": float(fit.inertia_), "cluster_sizes": np.bincount(fit.labels_, minlength=4).tolist()}
    return models, report


def extract(rgb, legacy, prototypes, orders):
    times = {}
    t = time.perf_counter()
    h3, gradient = texture(rgb)
    dc, dn, q = distances(h3, legacy["templates"])
    times["real_texture_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    nulls = [distances(lbp_hist(permute(rgb, order), 3), legacy["templates"])[2] for order in orders]
    mean_null = np.mean(nulls, axis=0)
    times["permutation_texture_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    multi = {3: multi_distances(h3, prototypes["3"])}
    for radius in (1, 9):
        multi[radius] = multi_distances(lbp_hist(rgb, radius), prototypes[str(radius)])
    times["additional_scale_prototype_seconds"] = time.perf_counter() - t
    records = []
    for i in range(len(rgb)):
        r = {"tile": i, "q_real": float(q[i]), "dc_real": float(dc[i]), "dn_real": float(dn[i]),
             "gradient": float(gradient[i]), "q_null_mean": float(mean_null[i]),
             "a": float(q[i] - mean_null[i]),
             "q_multi1": float(multi[1][i]), "q_multi3": float(multi[3][i]),
             "q_multi9": float(multi[9][i]),
             "q_multiscale": float(np.mean([multi[r][i] for r in RADII]))}
        r.update({f"q_null{k}": float(nulls[k][i]) for k in range(K)})
        records.append(r)
    return records, times


def feature_matrix(rows, keys):
    return np.asarray([[float(r[key]) for key in keys] for r in rows], dtype=np.float64)


def source_probability(rows, fit):
    x = feature_matrix(rows, fit["features"])
    z = (x - np.asarray(fit["mean"])) / np.asarray(fit["scale"])
    eta = z @ np.asarray(fit["coef"]) + fit["intercept"]
    return np.exp(-np.logaddexp(0, -eta))


def fit_mappings(source):
    population = [r for r in source if float(r["pred_cloud_frac"]) >= 0.5]
    truth = np.asarray([float(r["label_cloud_frac"]) < 0.5 for r in population], dtype=int)
    assert len(population) == 1034 and truth.sum() == 341
    fits = {}
    with threadpool_limits(limits=8):
        for name, keys in FEATURES.items():
            x = feature_matrix(population, keys)
            mean, scale = x.mean(0), x.std(0)
            scale = np.where(scale > 0, scale, 1.0)
            fit = LogisticRegression(C=1, solver="lbfgs", tol=1e-8, max_iter=200,
                                     random_state=SEED).fit((x - mean) / scale, truth)
            assert fit.n_iter_[0] < 200, (name, fit.n_iter_)
            entry = {"features": list(keys), "mean": mean.tolist(), "scale": scale.tolist(),
                     "coef": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0]),
                     "iterations": int(fit.n_iter_[0]), "n_cloud_calls": len(population),
                     "n_errors": int(truth.sum()), "n_parameters": len(keys) + 1}
            assert np.allclose(source_probability(population, entry), fit.predict_proba((x - mean) / scale)[:, 1], atol=1e-12)
            fits[name] = entry
    return fits
