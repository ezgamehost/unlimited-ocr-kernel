"""Isolate activation precision or packed storage on the failing hybrid case."""
import argparse
import json
from pathlib import Path
import subprocess

from tools.benchmark import benchmark_environment, validate_image
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['weight-only', 'simulation'], default='weight-only')
    args = parser.parse_args()
    original = json.loads((ROOT / 'head-major/process.json').read_text())
    command = original['command'].copy()
    assert command.count('--int8-activations') == 1
    if args.mode == 'weight-only':
        command.remove('--int8-activations')
        overlay_sha = '6981387fed4849dca9a713a717e9b54c04344adbaeff05410528773912e287ae'
        purpose = 'Activation precision diagnostic on the same hybrid weights; not a performance comparison.'
    else:
        command[command.index('--quantized-weights') + 1] = 'artifacts/unlimited-ocr-q4-q8head-sim.safetensors'
        overlay_sha = '741b40e3adf06ad2abb05c593e13a184303fd7d820142de15c8be6d268175f6b'
        purpose = 'Same hybrid codes/scales in physical INT8 storage; packing diagnostic, not performance comparison.'
    assert sha256(Path(command[0])) == original['binary_sha256']
    validate_image(Path(command[command.index('--image') + 1]), original['input_sha256'])
    assert sha256(Path(command[command.index('--quantized-weights') + 1])) == overlay_sha
    output = ROOT / args.mode
    output.mkdir(exist_ok=False)
    metadata = dict(command=command, binary_sha256=original['binary_sha256'],
                    input_sha256=original['input_sha256'],
                    power=subprocess.check_output(['pmset', '-g', 'batt'], text=True),
                    purpose=purpose, overlay_sha256=overlay_sha)
    with (output / 'result.json').open('wb') as out, (output / 'stderr.log').open('wb') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=benchmark_environment(8, 8), start_new_session=True)
        metadata['resources'] = wait_child(child, 900)
    (output / 'process.json').write_text(json.dumps(metadata, indent=2) + '\n')
    assert metadata['resources']['exit_code'] == 0 and not metadata['resources']['forced_kill']
    result = json.loads((output / 'result.json').read_text())
    truth = normalize_ocr((ROOT.parent / '2026-09-28-max32/portrait32.txt').read_text())
    text = normalize_ocr(result['text'])
    metadata.update(finish_reason=result['finish_reason'], tokens=len(result['token_ids']),
                    image_tokens=result['image_tokens'], cer=character_error_rate(truth, text))
    (output / 'transcript.txt').write_text(text)
    (output / 'process.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k:metadata[k] for k in ['finish_reason','tokens','image_tokens','cer','resources']}), flush=True)


if __name__ == '__main__':
    main()
