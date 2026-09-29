"""Compare one native prefill trajectory with frozen upstream configurations."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def metric(actual, reference):
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    error = (actual.double() - reference.double()).abs()
    outside = error > 1e-4 + 1e-4 * reference.double().abs()
    return {
        'max_abs': error.max().item(),
        'rms': error.square().mean().sqrt().item(),
        'outside': outside.sum().item(),
        'outside_indices': outside.nonzero().flatten().tolist(),
    }


def main():
    root = Path('artifacts/reference-thread-sensitivity')
    native_path = root / 'native-prefill.json'
    native = {k: torch.tensor(v, dtype=torch.float32)
              for k, v in json.loads(native_path.read_text()).items()}
    paths = {'original': Path('artifacts/reference-logo-f32/decoder.safetensors'),
             'threads10': root / 'prefill-threads-10.safetensors',
             'threads1': root / 'prefill-threads-1.safetensors'}
    result = {}
    for label, path in paths.items():
        ref = load_file(path)
        assert ref['logits_positions'][0].item() == 276
        assert native['logits'].shape == (129280,)
        result[label] = {
            'logits': metric(native['logits'], ref['logits'][0]),
            'same_top1': native['logits'].argmax().item() == ref['logits'][0].argmax().item(),
            'last_hidden_by_layer': {
                str(i): metric(native[f'layer.{i}.last_hidden'], ref[f'layer.{i}.hidden'][0, 276])
                for i in range(12)
            },
        }
    failures = [set(result[k]['logits']['outside_indices']) for k in paths]
    result['failed_in_all_references'] = sorted(set.intersection(*failures))
    result['sha256'] = {
        str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
        for p in [native_path, *paths.values(), Path(__file__)]
    }
    with Path(__file__).with_name('results.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    for label in paths:
        print(label, result[label]['logits'])
    print('Failed in all references:', result['failed_in_all_references'])


if __name__ == '__main__':
    main()
