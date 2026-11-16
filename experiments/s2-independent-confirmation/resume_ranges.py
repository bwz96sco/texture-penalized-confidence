"""Recover observed HTTP/2 cancellations without repeating received payload."""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data, execution, fixed, load


def main(task):
    started = time.perf_counter()
    freeze, _ = fixed(task)
    target = Path(freeze['cloudsen_file'])
    directory = target.parent / 'ranges'
    plan = load(directory / 'plan.json')
    originals = [load(Path(part['file'] + '.json')) for part in plan['ranges']]
    total = plan['total_bytes']
    jobs = []
    for part in plan['ranges']:
        path = Path(part['file'])
        start = part['start'] + path.stat().st_size
        for number, first in enumerate(range(start, part['end'] + 1, 200_000_000)):
            jobs.append({'parent': part['file'], 'start': first,
                         'end': min(first + 200_000_000 - 1, part['end']),
                         'file': str(directory / f'{path.name}-recovery-{number}')})
    assert target.stat().st_size == plan['prefix_bytes']

    def fetch(job):
        path = Path(job['file'])
        headers = Path(str(path) + '.headers')
        assert not path.exists(), 'Retain earlier recovery attempts; resume explicitly.'
        argv = ['curl', '--http1.1', '--location', '--fail', '--silent', '--show-error',
                '--connect-timeout', '30', '--max-time', '1800', '--range',
                f"{job['start']}-{job['end']}", '--dump-header', str(headers),
                '--output', str(path), '--write-out',
                '{"http_code":%{http_code},"size_download":%{size_download},"time_total":%{time_total},"speed_download":%{speed_download}}',
                freeze['url'] + '?download=true&confirmation_recovery=' + path.name]
        run = subprocess.run(argv, capture_output=True, text=True)
        native = json.loads(run.stdout)
        record = {**job, 'command': argv, 'native': native, 'exit_code': run.returncode,
                  'stderr': run.stderr, 'file_bytes': path.stat().st_size if path.exists() else 0}
        data.dump(Path(str(path) + '.json'), record)
        expected = f"content-range: bytes {job['start']}-{job['end']}/{total}"
        assert run.returncode == 0 and native['http_code'] == 206, record
        assert path.stat().st_size == job['end'] - job['start'] + 1
        assert expected in headers.read_text().lower()
        print({'recovered': path.name, 'bytes': path.stat().st_size, 'seconds': native['time_total']}, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        recovered = list(pool.map(fetch, jobs))
    for part in plan['ranges']:
        path = Path(part['file'])
        with path.open('ab') as dst:
            for item in [r for r in recovered if r['parent'] == str(path)]:
                with Path(item['file']).open('rb') as src:
                    shutil.copyfileobj(src, dst, 1 << 20)
        assert path.stat().st_size == part['end'] - part['start'] + 1
        expected = f"content-range: bytes {part['start']}-{part['end']}/{total}"
        assert expected in Path(str(path) + '.headers').read_text().lower()
    complete = Path(str(target) + '.complete')
    with complete.open('wb') as dst:
        for source in [target] + [Path(part['file']) for part in plan['ranges']]:
            with source.open('rb') as src:
                shutil.copyfileobj(src, dst, 1 << 20)
    assert complete.stat().st_size == total
    os.replace(complete, target)
    initial = load(task / 'evidence/transfer.json')
    measured = initial['measured_payload_bytes'] + sum(r['native']['size_download'] for r in originals + recovered)
    assert measured == total
    # Temporary recovery tails can be removed after their verified bytes are
    # retained in the parent ranges and complete object, keeping peak <=6 GB.
    for item in recovered:
        Path(item['file']).unlink()
    data.dump(task / 'evidence/transfer.json', {
        'transport': 'Original prefix, four disjoint ranges, and HTTP/1.1 missing-tail recovery; every 206 range and length checked.',
        'initial_single_stream': initial, 'prefix_bytes': plan['prefix_bytes'],
        'ranges': originals, 'recovery_ranges': recovered,
        'measured_payload_bytes': measured, 'image_file_bytes': total,
        'complete': True, 'assembly_python_isolated': True,
        'peak_payload_storage_bytes': 2 * total + sum(r['file_bytes'] for r in recovered),
        'execution': execution(started)})
    print({'assembled_bytes': total, 'unique_image_payload_bytes': measured, 'complete': True}, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
