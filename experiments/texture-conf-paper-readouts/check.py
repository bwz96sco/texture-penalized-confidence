"""Read-only verification: recompute the readouts (which assert the retained reproductions) and compare."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('readouts', Path(__file__).with_name('readouts.py'))
readouts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readouts)

task = Path(sys.argv[1]).resolve()
retained = json.loads((task / 'evidence/readouts.json').read_text())
fresh = json.loads(json.dumps(readouts.compute()))
assert fresh == retained, 'retained readouts differ from a fresh recomputation'
r0 = retained['R0_reproduction']
assert r0['untouched']['n_tiles'] == 80850 and r0['test_split']['n_tiles'] == 40425
print('Readouts reproduce; untouched clear calls accepted by texture-conf:',
      r0['untouched']['texture_conf_clear_calls_accepted'], 'of', r0['untouched']['n_clear_calls'])
