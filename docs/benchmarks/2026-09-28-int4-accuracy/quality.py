"""Compare original-prompt OCR quality with Q8 and unpacked Q4-level weights."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    items = []
    for path in (Path('benchmarks/corpus/manifest.json'), Path('benchmarks/heldout/manifest.json')):
        for item in json.loads(path.read_text())['images']:
            image = (path.parent / item['path']).resolve()
            validate_image(image, item['sha256'])
            items.append(dict(item, case=path.parent.name + '-' + item['id'], image=str(image)))
    binary = Path('target/release/uocr').resolve()
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = benchmark_environment(8, 8)
    variants = [('q8', 8, Path('artifacts/unlimited-ocr-q8-block32.safetensors')),
                ('q4-simulation', 4, Path('artifacts/unlimited-ocr-q4-sim-in-q8.safetensors'))]
    reports = {}
    for name, bits, overlay in variants:
        directory = ROOT / name
        directory.mkdir(exist_ok=False)
        manifest = json.loads(overlay.with_suffix('.json').read_text())
        assert manifest.get('simulated_weight_bits', 8) == bits and manifest.get('storage_bits', 8) == 8
        command = [str(binary), '--model', 'artifacts/upstream', '--jsonl', '--threads', '8',
                   '--int8-weights', str(overlay), '--int8-activations']
        requests = [dict(image=item['image'], prompt='<image>document parsing.',
                        generation=dict(max_tokens=8192, ngram_size=35, ngram_window=128)) for item in items]
        save(directory / 'requests.json', requests)
        process = dict(command=command, binary_sha256=binary_hash, overlay_manifest=manifest,
                       weight_level_bits=bits, storage_bits=8, cases=items,
                       purpose='Accuracy-only simulation; not packed INT4 performance.',
                       thread_environment={k: env[k] for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')})
        save(directory / 'process.json', process)
        start = time.monotonic()
        with (directory / 'responses.jsonl').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     env=env, start_new_session=True)
            with child.stdin as stream:
                for request in requests:
                    stream.write((json.dumps(request) + '\n').encode())
            process['resources'] = wait_child(child, 900)
        process['wall_seconds'] = time.monotonic() - start
        save(directory / 'process.json', process)
        assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
        responses = [json.loads(line) for line in (directory / 'responses.jsonl').read_text().splitlines()]
        assert len(responses) == len(items)
        reports[name] = {}
        for item, response in zip(items, responses):
            assert response['status'] == 'ok', response
            result = response['result']
            assert result['image_tokens'] == item.get('expected_image_tokens', 1513)
            row = dict(finish_reason=result['finish_reason'], emitted_tokens=len(result['token_ids']),
                       cer=character_error_rate(normalize_ocr(item['text']), normalize_ocr(result['text'])),
                       text=result['text'], token_ids=result['token_ids'])
            reports[name][item['case']] = row
            print(json.dumps(dict(variant=name, case=item['case'], cer=row['cer'],
                                  finish_reason=row['finish_reason'], emitted_tokens=row['emitted_tokens'])), flush=True)
        save(directory / 'quality.json', reports[name])
    comparison = {}
    for case, baseline in reports['q8'].items():
        candidate = reports['q4-simulation'][case]
        comparison[case] = dict(q8_cer=baseline['cer'], q4_cer=candidate['cer'],
            cer_delta=candidate['cer'] - baseline['cer'],
            both_eos=baseline['finish_reason'] == candidate['finish_reason'] == 'eos',
            q8_tokens=baseline['emitted_tokens'], q4_tokens=candidate['emitted_tokens'],
            exact_raw_output_match=baseline['text'] == candidate['text'] and baseline['token_ids'] == candidate['token_ids'])
    save(ROOT / 'comparison.json', dict(cases=comparison,
         note='All weights remain I8 storage; this report establishes neither packed-kernel speed nor memory savings.'))


if __name__ == '__main__':
    main()
