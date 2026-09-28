"""Exact Q8 and hybrid OCR gates after the final source-review fixes."""
import json
from pathlib import Path
import shutil
import subprocess

from tools.benchmark import benchmark_environment, validate_image, _stop_process_group
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent
BASELINE_SHA = 'c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    cli = Path('target/release/uocr')
    digest = sha256(cli)
    binary = Path('artifacts/kernel-baselines') / ('uocr-' + digest)
    if not binary.exists():
        shutil.copy2(cli, binary)
    assert sha256(binary) == digest
    save(ROOT / 'candidate-cli.json', dict(binary=str(binary), sha256=digest,
         source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         source_sha256={str(p): sha256(p) for p in sorted(Path('src').rglob('*.rs'))}))
    for name, baseline in [
        ('q8', ROOT.parent / '2026-09-28-current-q8/quality'),
        ('hybrid', ROOT.parent / '2026-09-28-kv-layout-probe/quality'),
    ]:
        old_process = json.loads((baseline / 'process.json').read_text())
        assert old_process['binary_sha256'] == BASELINE_SHA
        assert old_process['resources']['exit_code'] == 0
        assert not old_process['resources']['forced_kill']
        requests = json.loads((baseline / 'requests.json').read_text())
        cases = old_process['cases']
        old = [json.loads(line) for line in (baseline / 'responses.jsonl').read_text().splitlines()]
        assert len(requests) == len(cases) == len(old) == 12
        for case in cases:
            validate_image(Path(case['image']), case['sha256'])
        command = [str(binary.resolve()), *old_process['command'][1:]]
        overlay = Path(command[command.index('--quantized-weights') + 1])
        manifest = json.loads(overlay.with_suffix('.json').read_text())
        assert manifest == old_process['overlay_manifest']
        output = ROOT / name
        output.mkdir(exist_ok=False)
        save(output / 'requests.json', requests)
        process = dict(command=command, binary_sha256=digest, cases=cases,
                       overlay_manifest=manifest, overlay_sha256=sha256(overlay),
                       baseline=str(baseline), baseline_sha256=BASELINE_SHA,
                       baseline_responses_sha256=sha256(baseline / 'responses.jsonl'),
                       thread_environment=old_process['thread_environment'],
                       purpose='Exact-output correctness gate, not performance timing.')
        save(output / 'process.json', process)
        assert sha256(binary) == digest
        with (output / 'responses.jsonl').open('wb') as out, (output / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     env=benchmark_environment(8, 8), start_new_session=True)
            try:
                with child.stdin as stream:
                    for request in requests:
                        stream.write((json.dumps(request) + '\n').encode())
                process['resources'] = wait_child(child, 900)
            except BaseException:
                _stop_process_group(child)
                raise
        save(output / 'process.json', process)
        assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
        assert sha256(binary) == digest
        new = [json.loads(line) for line in (output / 'responses.jsonl').read_text().splitlines()]
        assert len(new) == 12 and all(r['status'] == 'ok' for r in [*old, *new])
        comparisons = []
        for case, before, after in zip(cases, old, new):
            a, b = before['result'], after['result']
            comparisons.append(dict(case=case['case'], exact_text=a['text'] == b['text'],
                exact_tokens=a['token_ids'] == b['token_ids'],
                eos=a['finish_reason'] == b['finish_reason'] == 'eos',
                image_tokens_equal=a['image_tokens'] == b['image_tokens'],
                precision_equal=a['precision'] == b['precision'], tokens=len(b['token_ids']),
                cer=character_error_rate(normalize_ocr(case['text']), normalize_ocr(b['text']))))
        save(output / 'comparison.json', comparisons)
        assert all(all(row[k] for k in ['exact_text', 'exact_tokens', 'eos', 'image_tokens_equal', 'precision_equal'])
                   for row in comparisons)
        print(json.dumps(dict(event='quality_complete', variant=name, exact_cases=12, binary_sha256=digest)), flush=True)


if __name__ == '__main__':
    main()
