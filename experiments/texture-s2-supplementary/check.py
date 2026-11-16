"""Read-only verification: recompute the frozen scoring and compare with the retained results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import FREEZE_SHA256, analyze

task = Path(sys.argv[1]).resolve()
retained = json.loads((task / 'evidence/results.json').read_text())
fresh = analyze(task)  # asserts freeze/input hashes and gate and zero-label mask and bootstrap parity
assert json.loads(json.dumps(fresh)) == retained, 'retained results differ from a fresh recomputation'
assert retained['freeze_sha256'] == FREEZE_SHA256 and retained['veto_threshold'] == 0.5
for name in ('cloudsen_A', 'cloudsen_B'):
    s = retained['settings'][name]
    assert s['n_tiles'] == 40425 and s['n_groups'] == 165
    assert {v['n_accepted'] for v in s['arms'].values()} == {32340}
for i in (1, 2):
    run = json.loads((task / f'evidence/runtime-{i}.json').read_text())
    assert run['cloud_call_texture_equals_all_tile_reference'] and not run['label_band_requested']
print('Texture S2 supplementary evaluation reproduces; decision:', retained['decision'])
