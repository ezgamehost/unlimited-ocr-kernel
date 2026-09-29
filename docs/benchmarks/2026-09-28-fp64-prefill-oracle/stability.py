"""Check the FP64 graph's own sensitivity to one versus ten CPU threads."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from probe import graph, metric


def main():
    torch.set_num_threads(1)
    fixture = load_file('artifacts/reference-thread-sensitivity/prefill-threads-10.safetensors')
    ten_path = Path('artifacts/fp64-prefill-oracle/prefill.safetensors')
    ten = load_file(ten_path)
    model = Path('artifacts/upstream/model-00001-of-000001.safetensors')
    assert hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() == '2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6'
    with safe_open(model, framework='pt', device='cpu') as weights:
        one = graph(weights, fixture['input_embeddings'], torch.float64)
    result = {}
    for name, actual in one.items():
        if actual.dtype == torch.int64:
            # Ordering is immaterial to routing set identity.
            assert torch.equal(actual.sort(-1).values, ten[name].sort(-1).values)
        else:
            result[name] = metric(actual, ten[name])
    result['sha256'] = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                        for p in (ten_path, Path(__file__), Path(__file__).with_name('probe.py'))}
    with Path(__file__).with_name('stability.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('FP64 one versus ten threads:', result['logits'])


if __name__ == '__main__':
    main()
