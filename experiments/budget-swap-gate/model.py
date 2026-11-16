"""Three fixed losses and label-free, equal-count conservative exchanges."""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from data import SEED, cut

L2 = .05
TEMPERATURE = .5
N_BOOT = 32
GUARD = 1.281551565545


def soft_weights(score):
    k = math.ceil(.8 * len(score))
    lo, hi = float(score.min()) - 40 * TEMPERATURE, float(score.max()) + 40 * TEMPERATURE
    for _ in range(80):
        threshold = (lo + hi) / 2
        if expit((score - threshold) / TEMPERATURE).sum() > k:
            lo = threshold
        else:
            hi = threshold
    return expit((score - (lo + hi) / 2) / TEMPERATURE)


def objective(kind, offset, x, error):
    d = x.shape[1]
    meta = {}
    if kind == 'pairwise':
        base = cut(offset)
        left, right = np.meshgrid(np.flatnonzero(base), np.flatnonzero(~base), indexing='ij')
        left, right = left.ravel(), right.ravel()
        sign = error[left].astype(float) - error[right]
        keep = sign != 0
        sign, dx = sign[keep], x[right[keep]] - x[left[keep]]
        db = offset[right[keep]] - offset[left[keep]]
        meta['n_opposite_cross_boundary_pairs'] = len(sign)
        def loss(theta):
            if not len(sign):
                return .5 * L2 * (theta @ theta), L2 * theta
            margin = sign * (db + dx @ theta)
            return (float(np.logaddexp(0, -margin).mean() + .5 * L2 * (theta @ theta)),
                    dx.T @ (-sign * expit(-margin)) / len(sign) + L2 * theta)
        return loss, d, meta
    if kind == 'soft-budget':
        def loss(theta):
            w = soft_weights(offset + x @ theta)
            a = w * (1 - w) / TEMPERATURE
            centered_error = error - (a @ error) / max(float(a.sum()), 1e-300)
            k = math.ceil(.8 * len(error))
            return (float((w @ error) / k + .5 * L2 * (theta @ theta)),
                    x.T @ (a * centered_error) / k + L2 * theta)
        return loss, d, meta
    assert kind == 'pointwise'
    def loss(params):
        theta, intercept = params[:d], params[d]
        score = offset + x @ theta + intercept
        correct = 1 - error
        residual = expit(score) - correct
        value = np.mean(np.logaddexp(0, score) - correct * score) + .5 * L2 * (theta @ theta)
        return float(value), np.r_[x.T @ residual / len(error) + L2 * theta, residual.mean()]
    return loss, d + 1, meta


def fit(kind, offset, x, error):
    fun, size, meta = objective(kind, offset, x, error)
    if kind == 'pointwise' and len(np.unique(error)) == 1:
        return {'theta': np.zeros(x.shape[1]).tolist(),
                'intercept': 30. if error[0] == 0 else -30.,
                'success': True, 'iterations': 0, 'message': 'one-class sample: no ranking correction',
                'loss': None, 'gradient_inf': None, 'n_labels': len(error),
                'n_errors': int(error.sum()), **meta}
    result = minimize(fun, np.zeros(size), jac=True, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 100})
    value, grad = fun(result.x)
    assert np.isfinite(result.x).all() and np.isfinite(grad).all()
    if not result.success:
        raise RuntimeError(f'{kind} optimizer did not converge: {result.message}')
    return {'theta': result.x[:x.shape[1]].tolist(),
            'intercept': float(result.x[-1]) if kind == 'pointwise' else 0.,
            'success': bool(result.success), 'iterations': int(result.nit),
            'message': str(result.message), 'loss': value,
            'gradient_inf': float(np.max(np.abs(grad))),
            'n_labels': len(error), 'n_errors': int(error.sum()), **meta}


def train(kind, offset, x, error, groups):
    model = fit(kind, offset, x, error)
    unique = sorted(set(groups))
    rng = np.random.default_rng(SEED)
    heads, draws = [], []
    for _ in range(N_BOOT):
        drawn = rng.choice(unique, size=len(unique), replace=True).tolist()
        idx = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
        heads.append(fit(kind, offset[idx], x[idx], error[idx]))
        draws.append(drawn)
    covariance = np.cov(np.asarray([m['theta'] for m in heads]), rowvar=False)
    return {'kind': kind, 'full_fit': model, 'slope_covariance': covariance.tolist(),
            'bootstrap_fits': heads, 'bootstrap_groups': draws,
            'n_bootstrap': N_BOOT, 'n_unique_calibration_groups': len(unique)}


def select(offset, x, sat, model, guard=GUARD):
    baseline = cut(offset)
    theta = np.asarray(model['full_fit']['theta'])
    covariance = np.asarray(model['slope_covariance'])
    score = offset + x @ theta
    old, new = np.flatnonzero(baseline), np.flatnonzero(~baseline)
    old = old[np.argsort(score[old], kind='stable')]
    new = new[np.argsort(-score[new], kind='stable')]
    n = min(len(old), len(new))
    old, new = old[:n], new[:n]
    dx = x[new] - x[old]
    margin = score[new] - score[old]
    variance = np.einsum('ij,ij->i', dx @ covariance, dx)
    passes = (margin > guard * np.sqrt(np.maximum(variance, 0))) & (sat[old] == sat[new])
    accepted = baseline.copy()
    accepted[old[passes]] = False
    accepted[new[passes]] = True
    assert accepted.sum() == baseline.sum()
    return accepted, {'n_pairs': n, 'n_swaps': int(passes.sum()),
                      'n_positive_mean_pairs': int((margin > 0).sum()),
                      'n_uncertain_positive_pairs': int(((margin > 0) & ~passes).sum())}


def smoke():
    rng = np.random.default_rng(8)
    x, offset = rng.normal(size=(20, 9)), rng.normal(size=20)
    error = np.asarray([0, 1] * 10)
    gradients = {}
    for kind in ('pairwise', 'soft-budget', 'pointwise'):
        fun, size, _ = objective(kind, offset, x, error)
        theta = rng.normal(scale=.1, size=size)
        analytic = fun(theta)[1]
        numerical = np.zeros(size)
        for j in range(size):
            step = np.eye(size)[j] * 1e-5
            numerical[j] = (fun(theta + step)[0] - fun(theta - step)[0]) / 2e-5
        delta = float(np.max(np.abs(numerical - analytic)))
        assert delta < 2e-6, (kind, delta)
        gradients[kind] = delta
    assert abs(soft_weights(offset).sum() - 16) < 1e-10
    zero = {'full_fit': {'theta': [0.] * 9}, 'slope_covariance': np.zeros((9, 9)).tolist()}
    mask, report = select(offset, x, np.zeros(20, bool), zero)
    assert np.array_equal(mask, cut(offset)) and report['n_swaps'] == 0
    # A known beneficial swap must preserve K, and a large uncertainty must block it.
    x = np.zeros((5, 9)); x[4, 0] = 1
    model = {'full_fit': {'theta': [10.] + [0.] * 8}, 'slope_covariance': np.zeros((9, 9)).tolist()}
    mask, _ = select(np.arange(5., 0, -1), x, np.zeros(5, bool), model)
    assert mask.tolist() == [True, True, True, False, True]
    model['slope_covariance'][0][0] = 100.
    mask, _ = select(np.arange(5., 0, -1), x, np.zeros(5, bool), model)
    assert mask.tolist() == [True, True, True, True, False]
    return {'finite_difference_max_errors': gradients, 'implicit_budget_gradient': 'pass',
            'equal_count_known_swap': 'pass', 'uncertainty_blocks_swap': 'pass',
            'zero_correction_recovers_baseline': 'pass'}
