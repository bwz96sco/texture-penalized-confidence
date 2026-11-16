"""Complete the same object with four disjoint ranges, as in the earlier room run."""
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
    prefix = target.stat().st_size
    total = freeze['expected_file_bytes']
    assert 0 < prefix < total
    directory = target.parent / 'ranges'
    directory.mkdir(exist_ok=False)
    span = (total - prefix + 3) // 4
    plan = [{'start': start, 'end': min(start + span, total) - 1,
             'file': str(directory / f'part-{i}')}
            for i, start in enumerate(range(prefix, total, span))]
    data.dump(directory / 'plan.json', {'prefix_bytes': prefix, 'total_bytes': total, 'ranges': plan})
    original = load(task / 'evidence/transfer.json')

    def fetch(part):
        path = Path(part['file'])
        headers = Path(str(path) + '.headers')
        argv = ['curl', '--location', '--fail', '--silent', '--show-error',
                '--connect-timeout', '30', '--max-time', '3600', '--range',
                f"{part['start']}-{part['end']}", '--dump-header', str(headers),
                '--output', str(path), '--write-out',
                '{"http_code":%{http_code},"size_download":%{size_download},"time_total":%{time_total},"speed_download":%{speed_download}}',
                freeze['url'] + '?download=true&confirmation_range=' + path.name]
        run = subprocess.run(argv, capture_output=True, text=True)
        native = json.loads(run.stdout)
        record = {**part, 'command': argv, 'native': native, 'exit_code': run.returncode,
                  'stderr': run.stderr, 'file_bytes': path.stat().st_size if path.exists() else 0}
        data.dump(directory / (path.name + '.json'), record)
        expected = f"content-range: bytes {part['start']}-{part['end']}/{total}"
        assert run.returncode == 0 and native['http_code'] == 206, record
        assert path.stat().st_size == part['end'] - part['start'] + 1
        assert expected in headers.read_text().lower(), 'Wrong HTTP byte range'
        print({'completed_range': path.name, 'bytes': path.stat().st_size,
               'seconds': native['time_total']}, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, plan))
    assert target.stat().st_size == prefix
    complete = Path(str(target) + '.complete')
    with complete.open('wb') as dst:
        for source in [target] + [Path(part['file']) for part in parts]:
            with source.open('rb') as src:
                shutil.copyfileobj(src, dst, 1 << 20)
    assert complete.stat().st_size == total
    os.replace(complete, target)
    data.dump(task / 'evidence/transfer.json', {
        'transport': 'Original prefix plus four disjoint ranges of the same pinned file; 206 content-range and sizes verified.',
        'initial_single_stream': original, 'prefix_bytes': prefix, 'ranges': parts,
        'measured_payload_bytes': original['measured_payload_bytes'] + sum(part['native']['size_download'] for part in parts),
        'image_file_bytes': total, 'complete': True, 'assembly_python_isolated': True,
        'peak_payload_storage_bytes': 2 * total,
        'execution': execution(started)})
    print({'assembled_bytes': total, 'complete': True}, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
