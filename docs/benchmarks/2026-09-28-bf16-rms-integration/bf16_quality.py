"""Validate scoped build against the completed accepted-runtime BF16 run."""
import json
import subprocess
from pathlib import Path

from tools.benchmark import benchmark_environment, validate_image, _stop_process_group
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    output = ROOT / 'bf16-quality'
    output.mkdir(exist_ok=False)
    baseline = ROOT.parent / '2026-09-28-pairwise-rms/bf16-quality'
    old_process = json.loads((baseline / 'baseline/process.json').read_text())
    assert old_process['binary_sha256'] == '05616a6d17bae60ceda0fefa365dcdb0f93b6987e00246cd8bb8979a7178ee5e'
    assert old_process['resources']['exit_code'] == 0 and not old_process['resources']['forced_kill']
    old = [json.loads(x)['result'] for x in (baseline / 'baseline/responses.jsonl').read_text().splitlines()]
    requests = json.loads((baseline / 'requests.json').read_text())
    cases = old_process['cases']
    assert len(requests) == len(cases) == len(old) == 12
    for case in cases:
        validate_image(Path(case['image']), case['sha256'])
    identity = json.loads((ROOT / 'quality/candidate-cli.json').read_text())
    binary = Path(identity['binary'])
    assert sha256(binary) == identity['sha256']
    command = [str(binary.resolve()), '--jsonl', '--threads', '8']
    save(output / 'requests.json', requests)
    process = dict(command=command, binary_sha256=identity['sha256'], cases=cases,
                   baseline_response_sha256=sha256(baseline / 'baseline/responses.jsonl'),
                   purpose='BF16 output equality; battery run, not performance evidence')
    with (output / 'responses.jsonl').open('xb') as out, (output / 'stderr.log').open('xb') as err:
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
    new = [json.loads(x) for x in (output / 'responses.jsonl').read_text().splitlines()]
    assert len(new) == 12 and all(x['status'] == 'ok' for x in new)
    comparisons = []
    for case, before, response in zip(cases, old, new):
        after = response['result']
        comparisons.append(dict(case=case['case'], exact_text=before['text'] == after['text'],
                                exact_tokens=before['token_ids'] == after['token_ids'],
                                eos=before['finish_reason'] == after['finish_reason'] == 'eos',
                                image_tokens_equal=before['image_tokens'] == after['image_tokens'],
                                precision=before['precision'] == after['precision'] == 'bf16_weights_fp32'))
    save(output / 'comparison.json', comparisons)
    assert all(all(v for k, v in row.items() if k != 'case') for row in comparisons)
    assert sha256(binary) == identity['sha256']
    print('All twelve scoped BF16 outputs exactly match accepted runtime', flush=True)


if __name__ == '__main__':
    main()
