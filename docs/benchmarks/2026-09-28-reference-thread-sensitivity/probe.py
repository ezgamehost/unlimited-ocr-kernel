"""Measure upstream FP32 prefill sensitivity; no native tolerance changes."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tools.reference import export_decoder


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.double() - expected.double()).abs()
    return {
        "max_abs": error.max().item(),
        "rms": error.square().mean().sqrt().item(),
        "unequal": (actual != expected).sum().item(),
        "violations_at_1e-4_plus_1e-4_abs_ref":
            (error > 1e-4 + 1e-4 * expected.double().abs()).sum().item(),
    }


@torch.inference_mode()
def main():
    output = Path('artifacts/reference-thread-sensitivity')
    output.mkdir(exist_ok=False)
    fixture = Path('artifacts/reference-logo-f32')
    metadata = json.loads((fixture / 'metadata.json').read_text())
    vision = load_file(fixture / 'vision.safetensors')['image_embeddings']
    original = load_file(fixture / 'decoder.safetensors')
    prefix = len(metadata['token_ids'])
    results = {}
    exports = {}
    for threads in (10, 1):
        torch.set_num_threads(threads)
        result = export_decoder(Path('artifacts/upstream'), vision, metadata,
                                steps=0, dtype=torch.float32)
        assert result['logits_positions'].tolist() == [prefix - 1]
        assert torch.equal(result['input_embeddings'], original['input_embeddings'][:, :prefix])
        save_file(result, output / f'prefill-threads-{threads}.safetensors')
        exports[threads] = result
        results[str(threads)] = {
            'logits_vs_original': compare(result['logits'], original['logits'][:1]),
            'layers_vs_original': {
                str(i): compare(result[f'layer.{i}.hidden'],
                                original[f'layer.{i}.hidden'][:, :prefix])
                for i in range(12)
            },
        }
    results['one_vs_ten'] = {
        'logits': compare(exports[1]['logits'], exports[10]['logits']),
        'layers': {str(i): compare(exports[1][f'layer.{i}.hidden'],
                                   exports[10][f'layer.{i}.hidden']) for i in range(12)},
    }
    results['identity'] = {
        'torch': torch.__version__, 'revision': metadata['revision'],
        'precision': 'Original BF16 weights expanded to FP32; unchanged upstream forwards',
        'scope': 'Prefill only; original fixture head has nine rows, this probe has one',
        'files_sha256': {
            str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
            for p in (fixture / 'decoder.safetensors', fixture / 'vision.safetensors',
                      Path('tools/reference.py'), Path(__file__))
        },
    }
    report = Path(__file__).with_name('results.json')
    with report.open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
    print(json.dumps(results['one_vs_ten'], indent=2), flush=True)


if __name__ == '__main__':
    main()
