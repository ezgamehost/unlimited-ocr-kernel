"""Require exact current Q8 OCR parity with the preserved pre-GELU/cache CLI."""
import json
from pathlib import Path
import subprocess

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    template = ROOT.parent / '2026-09-28-kv-layout-probe/quality'
    cases = json.loads((template / 'process.json').read_text())['cases']
    requests = json.loads((template / 'requests.json').read_text())
    assert len(cases) == len(requests) == 12
    for case in cases:
        validate_image(Path(case['image']), case['sha256'])
    overlay = Path('artifacts/unlimited-ocr-q8-block32.safetensors')
    assert sha256(overlay) == 'b6afd272e6f19de970f97245c83b09e110a20acc8033f2853d0e6739e5fde441'
    manifest = json.loads(overlay.with_suffix('.json').read_text())
    baseline_sha = '7baaad44d8ae6f56fd6f470d29271a0b0529be83a8eacd53b69d60c9140c08ba'
    current_sha = 'c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57'
    variants = [('baseline', Path('artifacts/kernel-baselines') / ('uocr-' + baseline_sha), baseline_sha),
                ('quality', Path('target/release/uocr'), current_sha)]
    outputs = []
    for name, binary, digest in variants:
        assert sha256(binary) == digest
        output = ROOT / name
        output.mkdir(exist_ok=False)
        command = [str(binary.resolve()), '--jsonl', '--threads', '8', '--quantized-weights', str(overlay), '--int8-activations']
        save(output / 'requests.json', requests)
        process = dict(command=command, binary_sha256=digest, cases=cases, overlay_manifest=manifest,
                       thread_environment={k:v for k,v in benchmark_environment(8,8).items() if k in ['OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
                       purpose='Twelve-case Q8 exact-output gate; not performance timing.')
        save(output / 'process.json', process)
        with (output / 'responses.jsonl').open('wb') as out, (output / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     env=benchmark_environment(8,8), start_new_session=True)
            with child.stdin as stream:
                for request in requests:
                    stream.write((json.dumps(request) + '\n').encode())
            process['resources'] = wait_child(child, 900)
        save(output / 'process.json', process)
        assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
        rows = [json.loads(line) for line in (output / 'responses.jsonl').read_text().splitlines()]
        assert len(rows) == 12 and all(r['status'] == 'ok' for r in rows)
        assert all(r['result']['precision'] == 'int8_weights_dynamic_activations' for r in rows)
        outputs.append(rows)
        print(json.dumps(dict(event='completed', variant=name)), flush=True)
    comparisons = []
    for case, old, new in zip(cases, *outputs):
        a,b = old['result'],new['result']
        comparisons.append(dict(case=case['case'], exact_text=a['text']==b['text'], exact_tokens=a['token_ids']==b['token_ids'],
            eos=a['finish_reason']==b['finish_reason']=='eos', image_tokens_equal=a['image_tokens']==b['image_tokens'],
            tokens=len(b['token_ids']), cer=character_error_rate(normalize_ocr(case['text']), normalize_ocr(b['text']))))
    save(ROOT / 'quality/comparison.json', comparisons)
    for row in comparisons:
        print(json.dumps(row), flush=True)
    assert all(all(row[k] for k in ['exact_text','exact_tokens','eos','image_tokens_equal']) for row in comparisons)


if __name__ == '__main__':
    main()
