"""Quality/resource diagnostic for the hybrid decoder at the 32-crop boundary."""
import json
from pathlib import Path
import subprocess
import time

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-09-28-max32'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    expected = json.loads((SOURCE / 'input.json').read_text())
    image = SOURCE / 'portrait32.png'
    validate_image(image, expected['sha256'])
    truth = normalize_ocr((SOURCE / 'portrait32.txt').read_text())
    variants = {
        'token-major': (Path('artifacts/kernel-baselines/uocr-2290e91db0a9304f8256a0a928e5dd9547a02e9206d67f2401696d5de98b5e29'),
                        '2290e91db0a9304f8256a0a928e5dd9547a02e9206d67f2401696d5de98b5e29'),
        'head-major': (Path('artifacts/kv-layout/uocr-c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57'),
                       'c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57'),
    }
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-block32.safetensors')
    assert sha256(overlay) == '6981387fed4849dca9a713a717e9b54c04344adbaeff05410528773912e287ae'
    outputs = []
    records = []
    for name, (binary, digest) in variants.items():
        assert sha256(binary) == digest
        directory = ROOT / name
        directory.mkdir(exist_ok=False)
        command = [str(binary.resolve()), '--model', 'artifacts/upstream', '--image', str(image),
                   '--prompt', '<image>document parsing.', '--threads', '8', '--max-tokens', '8192',
                   '--quantized-weights', str(overlay), '--int8-activations', '--json']
        power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        start = time.monotonic()
        with (directory / 'result.json').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, env=benchmark_environment(8, 8),
                                     start_new_session=True)
            resources = wait_child(child, 900)
        record = dict(variant=name, command=command, binary_sha256=digest, resources=resources,
                      power=power, wall_seconds=time.monotonic()-start, input_sha256=expected['sha256'],
                      note='Single fresh-process correctness/resource observation; not a performance comparison.')
        save(directory / 'process.json', record)
        assert resources['exit_code'] == 0 and not resources['forced_kill']
        result = json.loads((directory / 'result.json').read_text())
        assert result['image_tokens'] == expected['expected_image_tokens']
        transcript = normalize_ocr(result['text'])
        (directory / 'transcript.txt').write_text(transcript)
        record.update(finish_reason=result['finish_reason'], tokens=len(result['token_ids']),
                      cer=character_error_rate(truth, transcript))
        save(directory / 'process.json', record)
        records.append(record)
        outputs.append(result)
        print(json.dumps({k:record[k] for k in ['variant','finish_reason','tokens','cer','resources']}), flush=True)
    prior = ROOT.parent / '2026-09-28-max32-original-prompt'
    comparison = dict(exact_outputs={k:outputs[0][k] == outputs[1][k]
        for k in ['text','token_ids','finish_reason','image_tokens']}, observations=records,
        prior_transcript_matches={name:normalize_ocr(outputs[1]['text']) == normalize_ocr(json.loads(
            (prior / name / 'result.json').read_text())['text'])
            for name in ['native-original-bf16','native-original-dynamic-int8']})
    save(ROOT / 'comparison.json', comparison)
    assert all(comparison['exact_outputs'].values()), 'cache layout changes max32 OCR output'
    assert all(r['finish_reason'] == 'eos' for r in records), 'incomplete OCR; retain failure evidence'
    print(json.dumps(comparison['exact_outputs']), flush=True)


if __name__ == '__main__':
    main()
