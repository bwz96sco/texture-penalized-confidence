"""Resume the one pinned image object and retain native curl transfer evidence."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import data, execution, fixed


def main(task):
    started = time.perf_counter()
    freeze, _ = fixed(task)
    target = Path(freeze['cloudsen_file'])
    attempts = []
    for attempt in range(3):
        before = target.stat().st_size if target.exists() else 0
        if before == freeze['expected_file_bytes']:
            break
        assert before < freeze['expected_file_bytes']
        argv = ['curl', '--location', '--fail', '--silent', '--show-error',
                '--connect-timeout', '30', '--max-time', '7200', '--continue-at', '-',
                '--output', str(target), '--write-out',
                '{"http_code":%{http_code},"size_download":%{size_download},"time_total":%{time_total},"speed_download":%{speed_download}}',
                freeze['url'] + '?download=true']
        run = subprocess.run(argv, capture_output=True, text=True)
        after = target.stat().st_size if target.exists() else 0
        try:
            native = json.loads(run.stdout)
        except ValueError:
            native = {'stdout': run.stdout}
        attempts.append({'command': argv, 'exit_code': run.returncode, 'before_bytes': before,
                         'after_bytes': after, 'native': native, 'stderr': run.stderr})
        print({'attempt': attempt + 1, 'exit_code': run.returncode, 'file_bytes': after,
               'expected_bytes': freeze['expected_file_bytes']}, flush=True)
        data.dump(task / 'evidence/transfer.json', {'attempts': attempts,
            'image_file_bytes': after, 'measured_payload_bytes': sum(a['native'].get('size_download', 0) for a in attempts),
            'complete': after == freeze['expected_file_bytes'],
            'execution': execution(started)})
        if after == freeze['expected_file_bytes']:
            break
        if run.returncode not in (0, 18, 28, 56):
            raise RuntimeError(run.stderr)
    assert target.stat().st_size == freeze['expected_file_bytes'], 'Incomplete image object; keep it for resume.'


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
