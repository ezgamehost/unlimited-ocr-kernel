"""Complete receipt inference with the native product's original prompt."""
import hashlib
import argparse
import json
import subprocess
import time
from pathlib import Path

from tools.benchmark import benchmark_environment
from tools.benchmark_resident import wait_child
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=Path('benchmarks/heldout/receipt.png'))
    parser.add_argument('--truth', type=Path, default=Path('benchmarks/heldout/receipt.txt'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--precision', choices=['both', 'bf16', 'dynamic-int8'], default='both')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = Path('target/release/uocr').resolve()
    image = args.image
    truth = args.truth.read_text()
    env = benchmark_environment(8, 8)
    for name in (('bf16', 'dynamic-int8') if args.precision == 'both' else (args.precision,)):
        directory = args.output / ('native-original-' + name)
        directory.mkdir(exist_ok=False)
        cmd = [str(binary), '--model', 'artifacts/upstream', '--image', str(image),
               '--threads', '8', '--max-tokens', '8192', '--json']
        if name == 'dynamic-int8':
            cmd += ['--int8-weights', 'artifacts/unlimited-ocr-q8-block32.safetensors', '--int8-activations']
        record = dict(command=cmd, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                      prompt='native default: <image>document parsing.',
                      thread_environment={k: env[k] for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')})
        start = time.monotonic()
        with (directory / 'result.json').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, start_new_session=True)
            record['resources'] = wait_child(child, 300)
        record['wall_seconds'] = time.monotonic() - start
        (directory / 'process.json').write_text(json.dumps(record, indent=2) + '\n')
        assert record['resources']['exit_code'] == 0 and not record['resources']['forced_kill']
        result = json.loads((directory / 'result.json').read_text())
        record.update(finish_reason=result['finish_reason'], emitted_tokens=len(result['token_ids']),
                      cer=character_error_rate(normalize_ocr(truth), normalize_ocr(result['text'])))
        (directory / 'process.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
