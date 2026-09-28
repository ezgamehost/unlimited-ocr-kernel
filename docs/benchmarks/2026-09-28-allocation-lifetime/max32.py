"""Fresh-process resource check on the known failing 32-crop OCR input."""
import hashlib
import argparse
import json
import subprocess
import time
from pathlib import Path

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-09-28-max32'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    suffix = '-reverse' if args.reverse else ''
    identity = json.loads((ROOT / 'identity.json').read_text())
    expected = json.loads((SOURCE / 'input.json').read_text())
    image = SOURCE / 'portrait32.png'
    validate_image(image, expected['sha256'])
    results, outputs = {}, []
    env = benchmark_environment(8, 8)
    for name in (('candidate', 'base') if args.reverse else ('base', 'candidate')):
        directory = ROOT / ('max32-' + name + suffix)
        directory.mkdir(exist_ok=False)
        binary = Path('artifacts/uocr-lifetime-' + name).resolve()
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert digest == identity['binaries'][name]
        cmd = [str(binary), '--model', 'artifacts/upstream', '--image', str(image),
               '--prompt', '<image>\ndocument parsing.', '--threads', '8', '--max-tokens', '8192',
               '--int8-weights', 'artifacts/unlimited-ocr-q8-block32.safetensors',
               '--int8-activations', '--json']
        started = time.monotonic()
        with (directory / 'result.json').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, start_new_session=True)
            resources = wait_child(child, 300)
        record = dict(command=cmd, binary_sha256=digest, resources=resources,
                      wall_seconds=time.monotonic() - started, input_sha256=expected['sha256'],
                      thread_environment={k: env[k] for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')})
        save(directory / 'process.json', record)
        assert resources['exit_code'] == 0 and not resources['forced_kill']
        output = json.loads((directory / 'result.json').read_text())
        assert output['image_tokens'] == 3793
        record.update(finish_reason=output['finish_reason'], emitted_tokens=len(output['token_ids']),
                      cer=character_error_rate(normalize_ocr((SOURCE / 'portrait32.txt').read_text()),
                                               normalize_ocr(output['text'])))
        save(directory / 'process.json', record)
        results[name] = record
        outputs.append(output)
        print(json.dumps(record), flush=True)
    equal = all(outputs[0][k] == outputs[1][k] for k in ('text', 'token_ids', 'finish_reason', 'image_tokens'))
    save(ROOT / ('max32-comparison' + suffix + '.json'), dict(processes=results, exact_outputs_match=equal,
         note='Two fresh-process observations; token-limit OCR failure remains a failure, not successful latency.'))
    assert equal, 'candidate changes output'


if __name__ == '__main__':
    main()
