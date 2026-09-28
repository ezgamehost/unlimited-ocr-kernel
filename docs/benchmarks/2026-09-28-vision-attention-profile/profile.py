"""Capture warmed vision stacks only after the powered comparison is audited."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from tools.benchmark import benchmark_environment
from tools.benchmark_resident import LineReader, save, wait_child
from tools.fetch_model import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous captures')
    audit = json.loads((args.comparison / 'repeated-comparison.json').read_text())
    assert audit['audited_requests'] == 96 and audit['all_eos']
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    assert "Now drawing from 'AC Power'" in power, power
    baseline = args.comparison / 'repeated-0-native'
    process = json.loads((baseline / 'process.json').read_text())
    command = process['command']
    assert sha256(Path(command[0])) == process['binary_sha256']
    request_path = next(baseline.glob('*-english.request.json'))
    request = json.loads(request_path.read_text())
    expected = json.loads(request_path.with_suffix('').with_suffix('.raw.json').read_text())['result']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'request.json', request)
    metadata = dict(command=command, binary_sha256=process['binary_sha256'],
                    comparison=str(args.comparison.resolve()), power=power,
                    sampling_delay_seconds=0.2, sampling_duration_seconds=5,
                    note='Profile overhead included. Inspect stacks for stage attribution; aggregate thread samples are not wall-time percentages.')
    child = None
    failure = None
    with (args.output / 'process.stderr').open('wb') as stderr, \
            (args.output / 'process.stdout').open('wb') as stdout:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=stderr, env=benchmark_environment(8, 8),
                                     start_new_session=True, bufsize=0)
            deadline = time.monotonic() + 120
            while True:
                loaded = False
                for line in (args.output / 'process.stderr').read_bytes().splitlines():
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(event, dict) and event.get('event') == 'loaded':
                        metadata['load_event'] = event
                        loaded = True
                        break
                if loaded:
                    break
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('model failed to become ready')
                time.sleep(.1)
            reader = LineReader(child.stdout.fileno(), stdout)
            for label in ['warmup', 'profiled']:
                started = time.monotonic()
                child.stdin.write((json.dumps(request) + '\n').encode())
                if label == 'profiled':
                    time.sleep(metadata['sampling_delay_seconds'])
                    metadata['sample_start_after_request_seconds'] = time.monotonic() - started
                    capture = subprocess.run(['sample', str(child.pid), '5', '1', '-file',
                                              str((args.output / 'sample.txt').resolve())],
                                             capture_output=True, timeout=30)
                    metadata['sample_end_after_request_seconds'] = time.monotonic() - started
                    metadata['sample_exit_code'] = capture.returncode
                    (args.output / 'sample.stdout').write_bytes(capture.stdout)
                    (args.output / 'sample.stderr').write_bytes(capture.stderr)
                    capture.check_returncode()
                response = json.loads(reader.line(180))
                save(args.output / f'{label}.json', response)
                metadata[f'{label}_wall_seconds'] = time.monotonic() - started
                assert response['status'] == 'ok'
                result = response['result']
                assert result['finish_reason'] == 'eos'
                assert result['text'] == expected['text'] and result['token_ids'] == expected['token_ids']
                print(json.dumps(dict(event=label, timings=result['timings'])), flush=True)
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            raise
        finally:
            if child is not None:
                child.stdin.close()
                metadata['resources'] = wait_child(child, 30, terminate=failure is not None)
            metadata['failure'] = failure
            save(args.output / 'process.json', metadata)
    assert metadata['resources']['exit_code'] == 0
    assert not metadata['resources']['forced_kill']


if __name__ == '__main__':
    main()
