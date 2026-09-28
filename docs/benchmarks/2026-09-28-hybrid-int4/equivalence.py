"""Run packed Q4 and require exact OCR output equivalence to frozen simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child

ROOT = Path(__file__).resolve().parent
ORACLE = ROOT.parent / '2026-09-28-packed-int4-resident' / 'head-simulation-quality'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='packed-equivalence')
    args = parser.parse_args()
    output = ROOT / args.output_name
    output.mkdir(exist_ok=False)
    original = json.loads((ORACLE / 'process.json').read_text())
    requests = json.loads((ORACLE / 'requests.json').read_text())
    for item in original['cases']:
        validate_image(Path(item['image']), item['sha256'])
    binary = Path('target/release/uocr').resolve()
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-block32.safetensors')
    manifest = json.loads(overlay.with_suffix('.json').read_text())
    assert manifest['format'] == 'uocr-q4-q8head-block32-v1'
    command = [str(binary), '--jsonl', '--threads', '8', '--quantized-weights', str(overlay), '--int8-activations']
    env = benchmark_environment(8, 8)
    process = dict(command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   overlay_manifest=manifest, cases=original['cases'],
                   oracle_process=original, purpose='Packed versus simulation exact output gate; single run, not a speed benchmark.',
                   thread_environment={k: env[k] for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')})
    save(output / 'requests.json', requests)
    save(output / 'process.json', process)
    start = time.monotonic()
    with (output / 'responses.jsonl').open('wb') as out, (output / 'stderr.log').open('wb') as err:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, env=env, start_new_session=True)
        with child.stdin as stream:
            for request in requests:
                stream.write((json.dumps(request) + '\n').encode())
        process['resources'] = wait_child(child, 900)
    process['wall_seconds'] = time.monotonic() - start
    save(output / 'process.json', process)
    assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
    expected = [json.loads(line) for line in (ORACLE / 'responses.jsonl').read_text().splitlines()]
    actual = [json.loads(line) for line in (output / 'responses.jsonl').read_text().splitlines()]
    assert len(actual) == len(expected) == len(requests)
    rows = []
    for item, old, new in zip(original['cases'], expected, actual):
        assert old['status'] == new['status'] == 'ok'
        a, b = old['result'], new['result']
        row = dict(case=item['case'], exact_text=a['text'] == b['text'], exact_tokens=a['token_ids'] == b['token_ids'],
                   eos=a['finish_reason'] == b['finish_reason'] == 'eos', tokens=len(b['token_ids']),
                   image_tokens_equal=a['image_tokens'] == b['image_tokens'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    save(output / 'comparison.json', rows)
    assert all(all(row[key] for key in ('exact_text', 'exact_tokens', 'eos', 'image_tokens_equal')) for row in rows)


if __name__ == '__main__':
    main()
