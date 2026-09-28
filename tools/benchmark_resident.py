"""Matched resident CPU OCR measurements with raw streams and per-child resources."""
import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import random
import selectors
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.benchmark import (aggregate, benchmark_environment,
                             completion_text_without_eos, validate_image)
from tools.score import character_error_rate, normalize_ocr


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def schedule(images, warmups, repetitions, seed):
    if not images or len(set(images)) != len(images) or warmups < 0 or repetitions < 1:
        raise ValueError('invalid benchmark schedule')
    rng, jobs = random.Random(seed), []
    for iteration in range(-warmups, repetitions):
        order = list(images)
        rng.shuffle(order)
        jobs.extend(dict(image_id=image, iteration=iteration, warmup=iteration < 0)
                    for image in order)
    return jobs


class LineReader:
    def __init__(self, fd, log, limit=16 << 20):
        self.fd, self.log, self.limit, self.pending = fd, log, limit, bytearray()

    def line(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.fd, selectors.EVENT_READ)
            while True:
                end = self.pending.find(b'\n')
                if end >= 0:
                    result = bytes(self.pending[:end + 1])
                    del self.pending[:end + 1]
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('native response deadline')
                data = os.read(self.fd, 65536)
                if not data:
                    raise EOFError('native output ended before a complete response')
                self.log.write(data)
                self.log.flush()
                self.pending.extend(data)
                if len(self.pending) > self.limit:
                    raise ValueError('native response exceeds byte limit')


class StreamAccumulator:
    def __init__(self):
        self.pieces, self.tokens, self.final = [], [], None

    def push(self, line):
        if not line.startswith(b'data:'):
            return False
        body = line[5:].strip()
        if body == b'[DONE]':
            return False
        event = json.loads(body)
        if 'error' in event:
            raise ValueError(f'server stream error: {event["error"]}')
        if self.final is not None:
            raise ValueError('stream data after terminal event')
        piece, tokens = event.get('content', ''), event.get('tokens', [])
        self.pieces.append(piece)
        self.tokens.extend(tokens)
        if event.get('stop'):
            self.final = event
        return bool(piece or tokens)

    def finish(self):
        if self.final is None:
            raise ValueError('stream ended without a terminal event')
        if self.final['tokens_predicted'] != len(self.tokens):
            raise ValueError('stream token count does not match terminal metadata')
        return dict(self.final, content=''.join(self.pieces), tokens=self.tokens.copy())


def wait_child(process, timeout, terminate=False):
    """Reap this PID with wait4; never substitute cumulative child RSS."""
    if terminate:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline, forced = time.monotonic() + timeout, False
    while True:
        try:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        except ChildProcessError:
            return dict(exit_code=process.returncode, peak_rss_bytes=None,
                        resource_error='child was already reaped', forced_kill=forced)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return dict(exit_code=process.returncode, forced_kill=forced,
                        peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                        user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                        major_faults=usage.ru_majflt, minor_faults=usage.ru_minflt)
        if time.monotonic() >= deadline and not forced:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            forced = True
        time.sleep(.02)


def server_command(args, port):
    return [str(args.binary.resolve()), '-m', str(args.model.resolve()),
            '--mmproj', str(args.mmproj.resolve()), '-ngl', '0', '--device', 'none',
            '--no-mmproj-offload', '--no-op-offload', '--no-kv-offload',
            '-t', str(args.threads), '-tb', str(args.threads), '-c', '32768',
            '-n', str(args.max_tokens), '--temp', '0', '--seed', '42', '--repeat-penalty', '1',
            '--no-warmup', '--special', '--jinja', '-lv', '4', '--host', '127.0.0.1',
            '--port', str(port), '--parallel', '1', '--no-cache-prompt', '--cache-ram', '0',
            '--no-cache-idle-slots', '--no-context-shift', '--offline']


def http_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def run(args):
    env = benchmark_environment(args.threads, args.blas_threads)
    manifest = json.loads(args.manifest.read_text())
    images = {item['id']: item for item in manifest['images']}
    if len(images) != len(manifest['images']):
        raise ValueError('duplicate image IDs')
    for item in images.values():
        item['resolved_path'] = str((args.manifest.parent / item['path']).resolve())
        validate_image(Path(item['resolved_path']), item['sha256'])
    jobs = schedule(list(images), args.warmups, args.repetitions, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    port = None
    if args.engine == 'llama':
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        command = server_command(args, port)
    else:
        command = [str(args.binary.resolve()), '--model', str(args.model.resolve()), '--jsonl',
                   '--threads', str(args.threads)]
        if args.int8_weights:
            command += ['--int8-weights', str(args.int8_weights.resolve())]
        if args.int8_activations:
            command += ['--int8-activations']
    metadata = dict(engine=args.engine, command=command, process_mode='resident-independent-requests',
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    system=platform.platform(), schedule=jobs, max_tokens=args.max_tokens,
                    thread_environment={k: env[k] for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')},
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    warmups_per_image=args.warmups, measured_per_image=args.repetitions,
                    seed=args.seed, manifest=manifest, prefix_tokens=1518,
                    native_prompt='<image>\ndocument parsing.',
                    attention_note='native preserved prefix plus 128 generated; llama full attention',
                    first_token_note='native internal clock; llama first streamed content/token arrival')
    save(args.output / 'process.json', metadata)
    records, failure, child = [], None, None
    started = time.monotonic()
    with (args.output / 'process.stderr').open('wb') as stderr, \
            (args.output / 'process.stdout').open('wb') as stdout:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE if args.engine == 'native' else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE if args.engine == 'native' else stdout,
                                     stderr=stderr, env=env, start_new_session=True, bufsize=0)
            deadline = started + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError('inference process exited during startup')
                if args.engine == 'native':
                    lines = (args.output / 'process.stderr').read_bytes().splitlines()
                    loaded = None
                    for line in lines:
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if isinstance(event, dict) and event.get('event') == 'loaded':
                            loaded = event
                            break
                    if loaded:
                        metadata['load_event'] = loaded
                        break
                else:
                    try:
                        if http_json(f'http://127.0.0.1:{port}/health', 1).get('status') == 'ok':
                            break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('inference process startup deadline')
                time.sleep(.1)
            metadata['startup_wall_seconds'] = time.monotonic() - started
            if args.engine == 'llama':
                props = http_json(f'http://127.0.0.1:{port}/props', 5)
                if not props['modalities']['vision']:
                    raise ValueError('server has no vision support')
                save(args.output / 'props.json', props)
            else:
                reader = LineReader(child.stdout.fileno(), stdout)
            save(args.output / 'process.json', metadata)
            print(json.dumps(dict(event='ready', engine=args.engine,
                                  startup_seconds=metadata['startup_wall_seconds'])), flush=True)
            for index, job in enumerate(jobs):
                item = images[job['image_id']]
                prefix = args.output / f'{index:03}-{job["image_id"]}'
                record = dict(job, index=index, image_sha256=item['sha256'], status='error',
                              finish_reason='error', first_token_seconds=None)
                stamp = time.monotonic()
                try:
                    if args.engine == 'native':
                        request = dict(image=item['resolved_path'], prompt=metadata['native_prompt'],
                                       generation=dict(max_tokens=args.max_tokens))
                        save(prefix.with_suffix('.request.json'), request)
                        stamp = time.monotonic()
                        child.stdin.write((json.dumps(request) + '\n').encode())
                        raw = reader.line(args.timeout)
                        wall = time.monotonic() - stamp
                        prefix.with_suffix('.raw.json').write_bytes(raw)
                        response = json.loads(raw)
                        if response['status'] != 'ok':
                            raise ValueError(response.get('error', 'native inference failed'))
                        result = response['result']
                        text, tokens, timing = result['text'], result['token_ids'], result['timings']
                        record.update(finish_reason=result['finish_reason'], precision=result['precision'],
                                      first_token_seconds=timing['first_token_seconds'],
                                      decode_seconds=timing['decode_seconds'], decode_steps=timing['decode_steps'],
                                      vision_seconds=timing['vision_seconds'], prefill_seconds=timing['prefill_seconds'])
                        if result['image_tokens'] != 1513:
                            raise ValueError('unexpected image token count')
                    else:
                        payload = dict(prompt=dict(prompt_string=props['media_marker'] + 'document parsing.',
                                       multimodal_data=[base64.b64encode(Path(item['resolved_path']).read_bytes()).decode()]),
                                       temperature=0, seed=42, repeat_penalty=1, n_predict=args.max_tokens,
                                       cache_prompt=False, return_tokens=True, stream=True)
                        request = urllib.request.Request(f'http://127.0.0.1:{port}/completion',
                                  data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                        state = StreamAccumulator()
                        stamp = time.monotonic()
                        try:
                            with urllib.request.urlopen(request, timeout=args.timeout) as response, \
                                    prefix.with_suffix('.sse').open('wb') as raw:
                                for line in response:
                                    raw.write(line)
                                    raw.flush()
                                    generated = state.push(line)
                                    elapsed = time.monotonic() - stamp
                                    if generated and record['first_token_seconds'] is None:
                                        record['first_token_seconds'] = elapsed
                                    if elapsed > args.timeout:
                                        raise TimeoutError('server response deadline')
                                    if len(state.tokens) > args.max_tokens + 1:
                                        raise ValueError('server output exceeds cap')
                                    if state.final is not None:
                                        break
                        except urllib.error.HTTPError as error:
                            prefix.with_suffix('.http-error').write_bytes(error.read())
                            raise
                        wall = time.monotonic() - stamp
                        result = state.finish()
                        save(prefix.with_suffix('.raw.json'), result)
                        text, tokens = completion_text_without_eos(result, props['eos_token'])
                        timing = result['timings']
                        if timing['cache_n'] != 0 or timing['prompt_n'] != 1518:
                            raise ValueError('cached or mismatched prompt in server response')
                        finish = 'context_limit' if result['truncated'] else result['stop_type']
                        record.update(finish_reason='token_limit' if finish == 'limit' else finish,
                                      decode_seconds=timing['predicted_ms'] / 1000,
                                      decode_steps=max(0, timing['predicted_n'] - 1),
                                      prompt_with_vision_seconds=timing['prompt_ms'] / 1000,
                                      cache_tokens=timing['cache_n'], precision='Q8_0 decoder, F16 projector')
                    save(prefix.with_suffix('.decoded.json'), dict(text=text, token_ids=tokens))
                    record.update(status='ok', wall_seconds=wall, emitted_tokens=len(tokens), timings=timing,
                                  cer=character_error_rate(normalize_ocr(item['text']), normalize_ocr(text)))
                except Exception as error:
                    record.update(error=f'{type(error).__name__}: {error}', wall_seconds=time.monotonic() - stamp)
                    failure = record['error']
                records.append(record)
                save(prefix.with_suffix('.record.json'), record)
                print(json.dumps({k: record.get(k) for k in
                      ('image_id', 'iteration', 'warmup', 'status', 'finish_reason', 'wall_seconds', 'cer')}), flush=True)
                if failure:
                    break
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            if child is not None:
                if child.stdin:
                    try:
                        child.stdin.close()
                    except BrokenPipeError:
                        pass
                metadata['resources'] = wait_child(child, 20, terminate=args.engine == 'llama' or failure is not None)
                if child.stdout:
                    child.stdout.close()
                resources = metadata['resources']
                allowed_codes = (0, -signal.SIGTERM) if args.engine == 'llama' else (0,)
                if failure is None and (resources['exit_code'] not in allowed_codes
                                        or resources['forced_kill'] or not resources['peak_rss_bytes']):
                    failure = 'abnormal process termination or missing resource accounting'
            metadata['process_wall_seconds'] = time.monotonic() - started
            metadata['failure'] = failure
            save(args.output / 'process.json', metadata)
    measured = [r for r in records if not r['warmup']]
    protocol_complete = len(records) == len(jobs) and failure is None
    summary = dict(complete=protocol_complete and all(r['status'] == 'ok' and r['finish_reason'] == 'eos'
                                                    for r in records),
                   protocol_complete=protocol_complete,
                   expected_requests=len(jobs), collected_requests=len(records),
                   per_image={image: aggregate([r for r in measured if r['image_id'] == image]) for image in images})
    save(args.output / 'summary.json', summary)
    if failure:
        raise RuntimeError(failure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['native', 'llama'], required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mmproj', type=Path)
    parser.add_argument('--int8-weights', type=Path)
    parser.add_argument('--int8-activations', action='store_true')
    parser.add_argument('--manifest', type=Path, default=Path('benchmarks/corpus/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--blas-threads', type=int, default=8)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--startup-timeout', type=float, default=120)
    args = parser.parse_args()
    if args.engine == 'llama' and not args.mmproj:
        parser.error('llama requires --mmproj')
    if args.int8_activations and not args.int8_weights:
        parser.error('--int8-activations requires --int8-weights')
    if args.max_tokens < 1 or args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error('caps and timeouts must be positive')
    run(args)


if __name__ == '__main__':
    main()
