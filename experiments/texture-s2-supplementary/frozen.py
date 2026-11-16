"""Frozen texture configuration for the 165 independent S2 test ROIs: shared definitions.

Texture is needed only on predicted-cloud tiles: every scored arm multiplies the
cloud-call mapping by `main`. The class-count-matched control uses no texture.
"""
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/texture-permutation-premise'))
from perm_features import RADII, lbp_hist, multi_distances, source_probability  # noqa: E402

PREMISE = ROOT / 'tasks/texture-permutation-premise/evidence'
CONFIRMATION = ROOT / 'tasks/s2-independent-confirmation/evidence'
ZERO_LABEL = ROOT / 'tasks/zero-label-s2-test/evidence'
MAPPING = 'multi-scale-template/v1'
A_WEIGHT = 0.5  # zero-label rule, unchanged from tasks/zero-label-s2-test
VETO_THRESHOLD, VETO_SHIFT = 0.5, 1e3  # veto-zl, unchanged from tasks/texture-veto-midterm
SAT_DN, SAT_SHARE, SAT_SHIFT = 65535, 0.1, 1e6  # saturation term, as in experiments/cloudsen12-room
GATES = {'confidence': 'softmax-response-cloudsen/v1', 'ensemble': 'ensemble-entropy-cloudsen/v1'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut(score):
    accepted = np.zeros(len(score), bool)
    accepted[np.argsort(-np.asarray(score), kind='stable')[:math.ceil(.8 * len(score))]] = True
    return accepted


def q_multiscale(rgb, prototypes):
    """(N, 64, 64, 3) -> source-template multi-scale LBP score, as `multi-scale-template/v1`."""
    if not len(rgb):
        return np.zeros(0)
    return np.mean([multi_distances(lbp_hist(rgb, r), prototypes[str(r)]) for r in RADII], axis=0)


def p_cloud_call(q, mapping):
    return source_probability([{'q_multiscale': float(v)} for v in q], mapping) if len(q) else np.zeros(0)


def sat_fraction(dn_tiles):
    """(N, 64, 64, 4) raw DN -> share of pixels saturated in any band."""
    return (dn_tiles >= SAT_DN).any(-1).mean((1, 2))


def zero_label(conf, cloud_fraction, scale):
    zc = (conf - scale['mean'][0]) / scale['sd'][0]
    zf = (cloud_fraction - scale['mean'][1]) / scale['sd'][1]
    return A_WEIGHT * zc - (1 - A_WEIGHT) * zf


def matched(base, main, reference):
    """Keep `reference`'s accepted counts per predicted class; rank within each class by `base`."""
    accepted = np.zeros(len(base), bool)
    for cls in (False, True):
        idx = np.flatnonzero(main == cls)
        n = int((reference & (main == cls)).sum())
        accepted[idx[np.argsort(-base[idx], kind='stable')[:n]]] = True
    return accepted


def scores(conf, cloud_fraction, ens, sat_frac, p_cc, scale):
    """All frozen score arms; p_cc is 0 on predicted-clear tiles, where it is never computed."""
    main = cloud_fraction >= .5
    sat = (sat_frac > SAT_SHARE) & ~main
    zero = zero_label(conf, cloud_fraction, scale)
    veto = main & (p_cc > VETO_THRESHOLD)
    base = conf - SAT_SHIFT * sat
    return main, base, veto, {
        'confidence': conf, 'ensemble': ens, 'zero-label': zero,
        'veto-zl': zero - VETO_SHIFT * veto,
        'texture-conf': base - main * p_cc}


def masks(conf, cloud_fraction, ens, sat_frac, p_cc, scale):
    main, base, veto, score = scores(conf, cloud_fraction, ens, sat_frac, p_cc, scale)
    accepted = {arm: cut(s) for arm, s in score.items()}
    accepted['matched-control'] = matched(base, main, accepted['texture-conf'])
    return main, veto, accepted
