"""Read-only verification: recompute the frozen scoring on the untouched sample and compare."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import FREEZE_SHA256, analyze  # noqa: E402

task = Path(sys.argv[1]).resolve()
evidence = task / 'evidence'
retained = json.loads((evidence / 'results.json').read_text())
assert json.loads(json.dumps(analyze(task))) == retained, 'retained results differ from a fresh recomputation'
assert retained['freeze_sha256'] == FREEZE_SHA256
for name in ('cloudsen_A', 'cloudsen_B'):
    s = retained['settings'][name]
    assert s['n_tiles'] == 80850 and s['n_groups'] == 330
    assert {v['n_accepted'] for v in s['arms'].values()} == {64680}
runtime, truth = (json.loads((evidence / f).read_text()) for f in ('runtime.json', 'truth.json'))
assert runtime['freeze_sha256'] == FREEZE_SHA256 and not runtime['label_band_requested'] and truth['after_prediction_pass']
print('Texture S2 confirmation reproduces; decision:', retained['decision'])
