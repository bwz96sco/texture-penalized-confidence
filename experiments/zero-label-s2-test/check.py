"""Read-only verification: recompute the frozen evaluation and compare with the retained results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import A_WEIGHT, analyze

task = Path(sys.argv[1]).resolve()
retained = json.loads((task / 'evidence/results.json').read_text())
fresh = analyze()  # asserts mask and bootstrap parity with the recorded independent confirmation
assert retained['frozen_rule']['a_weight_conf'] == A_WEIGHT == 0.5
assert json.loads(json.dumps(fresh)) == retained, 'retained results differ from a fresh recomputation'
for name in ('cloudsen_A', 'cloudsen_B'):
    s = retained['settings'][name]
    assert s['n_tiles'] == 40425 and s['n_groups'] == 165
    assert len({v['n_accepted'] for v in s['arms'].values()}) == 1
print('Zero-label S2 test evaluation reproduces; recorded masks and bootstrap risks match; decision:', retained['decision'])
