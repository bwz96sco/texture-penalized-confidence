"""Selective risk, oracle risk, AUROC and the capture-clustered bootstrap."""
import math

import numpy as np
from scipy.stats import rankdata


def n_accept(n: int, coverage: float) -> int:
    return math.ceil(coverage * n)


def selective_risk(score: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Error rate among the ceil(coverage * N) best-scored tiles (stable order breaks ties)."""
    k = n_accept(len(score), coverage)
    s = np.where(np.isfinite(score), score, -np.inf)
    top = np.argsort(-s, kind="stable")[:k]
    return float(1 - correct[top].mean())


def oracle_risk(correct: np.ndarray, coverage: float) -> float:
    k = n_accept(len(correct), coverage)
    return max(0, k - int(correct.sum())) / k


def auroc(score: np.ndarray, correct: np.ndarray) -> float:
    """Probability that a correct tile outscores a wrong one (ties count half)."""
    pos, neg = int(correct.sum()), int((~correct).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    s = np.where(np.isfinite(score), score, -np.inf).astype(np.float64)  # float32 ranks lose precision
    r = rankdata(s)
    return float((r[correct].sum() - pos * (pos + 1) / 2) / (pos * neg))


def cluster_draws(n_clusters: int, draws: int, seed: int) -> np.ndarray:
    """Bootstrap draws of cluster indices: (draws, n_clusters)."""
    return np.random.default_rng(seed).integers(0, n_clusters, size=(draws, n_clusters))


def percentile_interval(values, level: float = 0.95):
    v = np.asarray(values, dtype=float)
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))
