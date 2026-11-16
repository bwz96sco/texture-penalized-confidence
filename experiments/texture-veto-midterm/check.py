"""Read-only verification: refit the source-only mapping, recompute the frozen scoring, compare with retained files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import VETO_THRESHOLD, analyze  # noqa: E402

task = Path(sys.argv[1]).resolve()
retained = json.loads((task / 'evidence/results.json').read_text())
retained_fit = json.loads((task / 'evidence/class-blind-fit.json').read_text())
fresh, fit = analyze()  # asserts count and interval parity with the permutation premise
assert json.loads(json.dumps(fit)) == retained_fit, 'class-blind fit differs from the retained one'
assert json.loads(json.dumps(fresh)) == retained, 'retained results differ from a fresh recomputation'
assert retained['frozen']['veto_threshold'] == VETO_THRESHOLD == 0.5
for name, n in (('hypso', 3920), ('cloudsen_A', 26215), ('cloudsen_B', 26215)):
    s = retained['settings'][name]
    assert s['n_tiles'] == n and len({v['n_accepted'] for v in s['arms'].values()}) == 1
print('Texture veto mid-term reproduces; permutation-premise counts and interval match; decision:', retained['decision'])
