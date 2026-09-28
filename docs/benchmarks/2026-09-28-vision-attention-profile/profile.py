"""Capture warmed vision or decode stacks after an audited comparison."""
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
    parser.add_argument('--quality', type=Path, help='Exact 12-case quality proof for a newer binary')
    parser.add_argument('--stage', choices=['vision', 'decode'], default='vision')
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
    quality_process = None
    expected_binary = process['binary_sha256']
    if args.quality:
        comparison = json.loads((args.quality / 'comparison.json').read_text())
        assert len(comparison) == 12 and all(all(row[key] for key in
            ['exact_text', 'exact_tokens', 'eos', 'image_tokens_equal']) for row in comparison)
        quality_process = json.loads((args.quality / 'process.json').read_text())
        assert quality_process['resources']['exit_code'] == 0
        assert not quality_process['resources']['forced_kill']
        expected_binary = quality_process['binary_sha256']
    assert sha256(Path(command[0])) == expected_binary
    page = 'dense' if args.stage == 'decode' else 'english'
    request_path = next(baseline.glob(f'*-{page}.request.json'))
    request = json.loads(request_path.read_text())
    if quality_process:
        assert any(case['image'] == request['image'] and case['prompt'] == request['prompt']
                   for case in quality_process['cases'])
    expected = json.loads(request_path.with_suffix('').with_suffix('.raw.json').read_text())['result']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'request.json', request)
    metadata = dict(command=command, binary_sha256=expected_binary, quality_process=quality_process,
                    comparison=str(args.comparison.resolve()), power=power,
                    stage=args.stage, sampling_delay_seconds=0.2,
                    sampling_duration_seconds=3 if args.stage == 'decode' else 5,
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
                    capture = subprocess.run(['sample', str(child.pid), str(metadata['sampling_duration_seconds']), '1', '-file',
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
                if label == 'warmup' and args.stage == 'decode':
                    metadata['sampling_delay_seconds'] = result['timings']['first_token_seconds'] + 0.15
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
