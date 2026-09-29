"""Matched BF16 output-quality comparison; timings are not benchmark evidence."""
import json
import subprocess
from pathlib import Path

from tools.benchmark import benchmark_environment, validate_image, _stop_process_group
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256
from tools.score import normalize_ocr, character_error_rate

ROOT = Path(__file__).resolve().parent
BASELINE = '05616a6d17bae60ceda0fefa365dcdb0f93b6987e00246cd8bb8979a7178ee5e'
CANDIDATE = '7561f20d042ca04a048a676462f7fa978f8a9b17909a477063c05b100c7659eb'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    output = ROOT / 'bf16-quality'
    output.mkdir(exist_ok=False)
    old = ROOT.parent / '2026-09-28-current-q8/quality'
    cases = json.loads((old / 'process.json').read_text())['cases']
    requests = json.loads((old / 'requests.json').read_text())
    assert len(cases) == len(requests) == 12
    for case in cases:
        validate_image(Path(case['image']), case['sha256'])
    save(output / 'requests.json', requests)
    completed = {}
    for label, digest in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
        binary = Path('artifacts/kernel-baselines') / ('uocr-' + digest)
        assert sha256(binary) == digest
        folder = output / label
        folder.mkdir()
        command = [str(binary.resolve()), '--jsonl', '--threads', '8']
        process = dict(command=command, binary_sha256=digest, cases=cases,
                       purpose='BF16 output quality only; battery execution, no speed claim')
        save(folder / 'launch.json', process)
        with (folder / 'responses.jsonl').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
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
        save(folder / 'process.json', process)
        assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
        assert sha256(binary) == digest
        responses = [json.loads(line) for line in (folder / 'responses.jsonl').read_text().splitlines()]
        assert len(responses) == 12 and all(x['status'] == 'ok' for x in responses)
        completed[label] = [x['result'] for x in responses]
        assert all(x['precision'] == 'bf16_weights_fp32' for x in completed[label])
        print(label, 'completed', flush=True)
    report = []
    for case, before, after in zip(cases, completed['baseline'], completed['candidate']):
        report.append(dict(case=case['case'], raw_equal=before['text'] == after['text'],
                           normalized_equal=normalize_ocr(before['text']) == normalize_ocr(after['text']),
                           tokens_equal=before['token_ids'] == after['token_ids'],
                           baseline_eos=before['finish_reason'] == 'eos', candidate_eos=after['finish_reason'] == 'eos',
                           baseline_cer=character_error_rate(normalize_ocr(case['text']), normalize_ocr(before['text'])),
                           candidate_cer=character_error_rate(normalize_ocr(case['text']), normalize_ocr(after['text']))))
    save(output / 'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
