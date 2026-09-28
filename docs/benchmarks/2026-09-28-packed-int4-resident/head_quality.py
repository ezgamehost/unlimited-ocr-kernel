"""Quality-only hybrid simulation: original eight pages plus four matched prompts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    out = ROOT / 'head-simulation-quality'
    out.mkdir(exist_ok=False)
    cases = []
    for manifest_path in [Path('benchmarks/corpus/manifest.json'), Path('benchmarks/heldout/manifest.json')]:
        for item in json.loads(manifest_path.read_text())['images']:
            image = (manifest_path.parent / item['path']).resolve()
            validate_image(image, item['sha256'])
            for prompt_name, prompt in [('original', '<image>document parsing.')] + ([('matched', '<image>\ndocument parsing.')] if manifest_path.parent.name == 'corpus' else []):
                cases.append(dict(item, image=str(image), case=manifest_path.parent.name + '-' + item['id'] + '-' + prompt_name, prompt=prompt))
    requests = [dict(image=c['image'], prompt=c['prompt'], generation=dict(max_tokens=8192, ngram_size=35, ngram_window=128)) for c in cases]
    binary = Path('target/release/uocr').resolve()
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-sim.safetensors')
    command = [str(binary), '--jsonl', '--threads', '8', '--int8-weights', str(overlay), '--int8-activations']
    process = dict(command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   overlay_manifest=json.loads(overlay.with_suffix('.json').read_text()), cases=cases,
                   purpose='Quality-only Q4 plus Q8 output-head simulation in I8 storage; no speed claim.')
    save(out / 'process.json', process); save(out / 'requests.json', requests)
    start = time.monotonic()
    with (out / 'responses.jsonl').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=benchmark_environment(8, 8), start_new_session=True)
        with child.stdin as stream:
            for request in requests:
                stream.write((json.dumps(request) + '\n').encode())
        process['resources'] = wait_child(child, 900)
    process['wall_seconds'] = time.monotonic() - start
    save(out / 'process.json', process)
    assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
    responses = [json.loads(line) for line in (out / 'responses.jsonl').read_text().splitlines()]
    assert len(responses) == len(cases)
    rows = []
    for case, response in zip(cases, responses):
        assert response['status'] == 'ok'
        result = response['result']
        row = dict(case=case['case'], finish_reason=result['finish_reason'], tokens=len(result['token_ids']),
                   cer=character_error_rate(normalize_ocr(case['text']), normalize_ocr(result['text'])))
        rows.append(row); print(json.dumps(row), flush=True)
    save(out / 'quality.json', rows)
    assert all(row['finish_reason'] == 'eos' for row in rows)


if __name__ == '__main__':
    main()
