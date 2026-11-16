"""Check the actual descriptor change, preserved marginals and direct swap arithmetic."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perm_features import extract, lbp_hist, permutations, permute
from perm_analysis import swaps
from common import distances, dump, texture

rng = np.random.default_rng(41)
rgb = rng.uniform(.01, .8, (2, 64, 64, 3))
assert np.array_equal(lbp_hist(rgb, 3), texture(rgb)[0])
for radius in (1, 3, 9):
    h = lbp_hist(rgb, radius)
    assert np.allclose(h.sum(1), 1, atol=1e-12)
    assert np.array_equal(h, lbp_hist(np.rot90(rgb, axes=(1, 2)), radius))
    constant = lbp_hist(np.ones_like(rgb), radius)
    expected = np.zeros((2, 36)); expected[:, 35] = 1
    assert np.array_equal(constant, expected)
orders = permutations()
assert len(orders) == 4 and all(np.array_equal(np.sort(p), np.arange(4096)) for p in orders)
for order in orders:
    assert np.array_equal(np.sort(rgb.reshape(2, 4096, 3), axis=1),
                          np.sort(permute(rgb, order).reshape(2, 4096, 3), axis=1))
legacy = json.loads(Path('tasks/texture-full-diagnostic/evidence/freeze.json').read_text())
prototypes = {str(r): {key: [legacy['templates'][key]] * 4 for key in ('cloud', 'clear')} for r in (1, 3, 9)}
records, _ = extract(rgb, legacy, prototypes, orders)
q = distances(texture(rgb)[0], legacy['templates'])[2]
nulls = [distances(lbp_hist(permute(rgb, p), 3), legacy['templates'])[2] for p in orders]
for i, row in enumerate(records):
    assert abs(row['a'] - (q[i] - sum(n[i] for n in nulls)/4)) < 1e-12
constant, _ = extract(np.ones_like(rgb), legacy, prototypes, orders)
assert all(abs(r['a']) < 1e-12 for r in constant)
before = np.array([1,1,0,0,1,1,1], bool)
after = np.array([1,0,1,1,1,0,1], bool)
correct_before = np.array([0,1,1,0,0,1,1], bool)
correct_after = np.array([1,1,0,0,0,1,0], bool)
t = swaps(before, correct_before, after, correct_after)
assert (t['dropped_wrong'], t['refill_wrong'], t['common_fixed_output'], t['common_broken_output'], t['net_errors_removed']) == (0,2,1,1,-2)
dump(Path(sys.argv[1])/'evidence/smoke.json', {'synthetic_status':'pass', 'checks':[
    'radius-3 exact legacy descriptor parity', 'radii 1/3/9 normalization, rotation and constant-tile reference',
    'four bijective joint permutations preserve channel marginals', 'direct four-null contrast arithmetic',
    'constant-tile contrast zero', 'hand-counted acceptance and changed-output decomposition']})
print('Descriptor, permutation, contrast and hand-counted transition smoke checks passed.')
