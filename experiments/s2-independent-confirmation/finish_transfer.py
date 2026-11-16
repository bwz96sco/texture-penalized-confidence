"""Finish observed disconnected streams with short gap requests, preserving every prefix."""
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
    total = plan['total_bytes']
    assert target.stat().st_size == plan['prefix_bytes']

    def retained():
        paths = sorted(directory.glob('part-[0-9].json')) + sorted(directory.glob('part-[0-9]-recovery-[0-9].json')) + sorted(directory.glob('gap-*.json'))
        records = [load(path) for path in paths]
        fragments = [{'start': 0, 'end': plan['prefix_bytes'] - 1, 'file': str(target)}]
        for record in records:
            path = Path(record['file'])
            count = path.stat().st_size
            assert count == record['file_bytes'] == record['native']['size_download']
            if not count:
                continue
            assert record['native']['http_code'] == 206
            expected = f"content-range: bytes {record['start']}-{record['end']}/{total}"
            assert expected in Path(str(path) + '.headers').read_text().lower()
            fragments.append({'start': record['start'], 'end': record['start'] + count - 1, 'file': str(path)})
        return sorted(fragments, key=lambda r: r['start']), records

    fragments, records = retained()
    jobs, cursor = [], 0
    for fragment in fragments + [{'start': total, 'end': total}]:
        assert fragment['start'] >= cursor, 'Overlapping received payload'
        for first in range(cursor, fragment['start'], 20_000_000):
            last = min(first + 20_000_000, fragment['start']) - 1
            jobs.append({'start': first, 'end': last,
                         'file': str(directory / f'gap-{first}-{last}-{len(records)}')})
        cursor = fragment['end'] + 1
    print({'missing_bytes': sum(j['end'] - j['start'] + 1 for j in jobs), 'short_requests': len(jobs)}, flush=True)

    def fetch(job):
        path = Path(job['file'])
        assert not path.exists()
        argv = ['curl', '--http1.1', '--location', '--fail', '--silent', '--show-error',
                '--connect-timeout', '30', '--max-time', '300', '--range',
                f"{job['start']}-{job['end']}", '--dump-header', str(path) + '.headers',
                '--output', str(path), '--write-out',
                '{"http_code":%{http_code},"size_download":%{size_download},"time_total":%{time_total},"speed_download":%{speed_download}}',
                freeze['url'] + '?download=true&confirmation_gap=' + path.name]
        run = subprocess.run(argv, capture_output=True, text=True)
        native = json.loads(run.stdout)
        record = {**job, 'command': argv, 'native': native, 'exit_code': run.returncode,
                  'stderr': run.stderr, 'file_bytes': path.stat().st_size if path.exists() else 0}
        data.dump(Path(str(path) + '.json'), record)
        assert run.returncode == 0 and native['http_code'] == 206, record
        assert path.stat().st_size == job['end'] - job['start'] + 1
        print({'received_short_range': path.name, 'bytes': path.stat().st_size}, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, jobs))
    fragments, records = retained()
    cursor = 0
    for fragment in fragments:
        assert fragment['start'] == cursor
        cursor = fragment['end'] + 1
    assert cursor == total
    complete = Path(str(target) + '.complete')
    with complete.open('wb') as dst:
        for fragment in fragments:
            with Path(fragment['file']).open('rb') as src:
                shutil.copyfileobj(src, dst, 1 << 20)
    assert complete.stat().st_size == total
    os.replace(complete, target)
    initial = load(task / 'evidence/transfer.json')
    reported = sum(r['native']['size_download'] for r in records)
    accounted = plan['prefix_bytes'] + reported
    assert accounted == total
    data.dump(task / 'evidence/transfer.json', {
        'transport': 'One pinned object; every received prefix preserved, short missing-gap requests, verified 206 intervals partition the object exactly.',
        'initial_single_stream': initial, 'fragments': fragments[1:], 'attempts': records,
        'accounted_payload_lower_bound_bytes': accounted,
        'curl_reported_bytes_after_prefix': reported,
        'payload_accounting': 'Interrupted initial curl emitted no final counter; retained prefix bytes plus subsequent native counters. Protocol overhead and any discarded in-flight bytes are unmeasured.',
        'image_file_bytes': total, 'complete': True,
        'assembly_python_isolated': True, 'peak_payload_storage_bytes': 2 * total,
        'execution': execution(started)})
    print({'assembled_bytes': total, 'accounted_payload_lower_bound_bytes': accounted, 'complete': True}, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
