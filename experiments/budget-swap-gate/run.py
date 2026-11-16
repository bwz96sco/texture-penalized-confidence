"""Run the retained V1 episode, then V2 on exactly the same frozen evaluator."""
import csv
import datetime as dt
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from data import OLD, ROOT, SETTINGS, arrays, dump, errors, group, key, read_csv, roles, values
from evaluate import decisions, summarize
from model import train


def main():
    task, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    assert stage in ('v1', 'v2')
    evidence = task / 'evidence'
    started = time.perf_counter()
    rows = read_csv(OLD / 'tiles.csv')
    io_seconds = time.perf_counter() - started
    freeze = json.loads((evidence / 'freeze.json').read_text())
    splits = json.loads((evidence / 'splits.json').read_text())
    models = {} if stage == 'v1' else json.loads((evidence / 'v1-models.json').read_text())
    results, records, timing = {}, [], {}
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for setting, dataset, convention in SETTINGS:
        population = [r for r in rows if r['dataset'] == dataset]
        cal, held = roles(population, splits['datasets'][dataset])
        offset, x, main, sat = arrays(cal, freeze['scale'])
        error, _ = errors(cal, convention)
        groups = np.asarray([group(r) for r in cal])
        models.setdefault(setting, {})
        fit_started = time.perf_counter()
        for kind in (('pairwise', 'pointwise') if stage == 'v1' else ('soft-budget',)):
            models[setting][kind] = train(kind, offset, x, error, groups)
            models[setting][kind]['calibration_keys'] = [key(r) for r in cal]
            print(json.dumps({'stage': stage, 'setting': setting, 'fit': kind,
                              'n_labels': len(error), 'n_errors': int(error.sum())}), flush=True)
        fit_seconds = time.perf_counter() - fit_started
        eval_offset, eval_x, eval_main, eval_sat = arrays(held, freeze['scale'])
        eval_error, _ = errors(held, convention)
        eval_started = time.perf_counter()
        result, setting_records = summarize(held, eval_offset, eval_x, eval_main, eval_sat, eval_error,
            models[setting], n_draws=0 if stage == 'v1' else 2000,
            progress=lambda report: print(json.dumps({'stage': stage, 'setting': setting, **report}), flush=True))
        results[setting] = result
        records.extend([{'setting': setting, **r} for r in setting_records])
        timing[setting] = {'ground_fit_seconds': fit_seconds,
                           'evaluation_and_statistics_seconds': time.perf_counter() - eval_started,
                           'n_calibration_labels': len(error), 'n_held': len(held)}
        dump(evidence / ('v1-models.json' if stage == 'v1' else 'models.json'), models)
        print(json.dumps({'stage': stage, 'setting': setting,
            'risks': {name: arm['risk'] for name, arm in result['arms'].items()}}), flush=True)
    summary = {'stage': stage, 'budget': freeze['budget'],
               'interval_scope': 'conditional on fixed heads, covariances and commissioning split',
               'settings': results}
    if stage == 'v2':
        summary['decision'] = decisions(results)
        parity = {}
        for setting, dataset, convention in SETTINGS:
            pop = [r for r in rows if r['dataset'] == dataset]
            offset, _, _, _ = arrays(pop, freeze['scale'])
            err, _ = errors(pop, convention)
            from data import cut
            parity[setting] = int((cut(offset) & err).sum())
        assert parity == {'cloudsen_A': 4932, 'cloudsen_B': 4028, 'hypso': 265}
        summary['historical_full_population_baseline_errors'] = parity
        with (evidence / 'decisions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(records)
        # Local incremental deployment timing includes scoring, covariance and pairing.
        local = {}
        for setting, dataset, _ in SETTINGS:
            pop = [r for r in rows if r['dataset'] == dataset]
            _, held = roles(pop, splits['datasets'][dataset])
            offset, x, _, sat = arrays(held, freeze['scale'])
            from model import select
            elapsed = []
            for repeat in range(5):
                t = time.perf_counter()
                select(offset, x, sat, models[setting]['soft-budget'])
                if repeat >= 2:
                    elapsed.append(time.perf_counter() - t)
            local[setting] = {'n_tiles': len(held), 'median_complete_head_selection_seconds': float(np.median(elapsed)),
                              'microseconds_per_tile': float(1e6 * np.median(elapsed) / len(held))}
        timing['deployment_local'] = local
    dump(evidence / ('v1-results.json' if stage == 'v1' else 'results.json'), summary)
    execution = {'stage': stage, 'code_commit': code_commit,
                 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'total_seconds': time.perf_counter() - started, 'csv_io_seconds': io_seconds,
                 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                     (1024 ** 2 if platform.system() == 'Darwin' else 1024),
                 'timing': timing, 'raw_image_reads': 0, 'downloads': 0,
                 'numerical_threads': 1, 'new_base_or_CNN_fits': 0,
                 'model_bytes': (evidence / ('v1-models.json' if stage == 'v1' else 'models.json')).stat().st_size}
    dump(evidence / ('v1-execution.json' if stage == 'v1' else 'execution.json'), execution)
    print(json.dumps({'stage': stage, 'done_seconds': execution['total_seconds']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
