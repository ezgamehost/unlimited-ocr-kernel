"""Change one arithmetic policy at a time in the validated reference graph."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

source_path = Path('docs/benchmarks/2026-09-28-fp64-prefill-oracle/probe.py')
spec = importlib.util.spec_from_file_location('oracle', source_path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def native_order_norm(value, weight):
    # Match Rust's scalar f32 sum, divide, sqrt then reciprocal; no fused multiply-add.
    total = torch.zeros_like(value[..., :1])
    for column in range(value.shape[-1]):
        v = value[..., column:column + 1]
        total = total + v * v
    scale = (total / value.shape[-1] + 1e-6).sqrt().reciprocal()
    return value * scale * weight


def main():
    torch.set_num_threads(10)
    folder = Path(__file__).parent
    output = Path('artifacts/prefill-arithmetic-ablation')
    output.mkdir(exist_ok=False)
    ref = load_file('artifacts/reference-thread-sensitivity/prefill-threads-10.safetensors')
    precise = load_file('artifacts/fp64-prefill-oracle/prefill.safetensors')
    native = json.loads(Path('artifacts/reference-thread-sensitivity/native-prefill.json').read_text())
    native_logits = torch.tensor(native['logits'])[None]
    original = inspect.getsource(oracle.graph)
    norm_old = "return weight(name + '.weight') * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))"
    norm_new = "return native_order_norm(value, weight(name + '.weight'))"
    scale_old = 'q @ k.transpose(2, 3) / math.sqrt(128) + mask'
    scale_new = 'q @ k.transpose(2, 3) * torch.tensor(128, dtype=dtype).sqrt().reciprocal() + mask'
    assert original.count(norm_old) == original.count(scale_old) == 1
    results = {}
    for label, norm_change, scale_change in (
        ('unchanged_control', False, False), ('scalar_rms', True, False),
        ('reciprocal_attention_scale', False, True), ('both', True, True),
    ):
        source = original.replace(norm_old, norm_new) if norm_change else original
        source = source.replace(scale_old, scale_new) if scale_change else source
        (output / f'{label}.py').write_text(source)
        namespace = dict(oracle.__dict__, native_order_norm=native_order_norm)
        exec(compile(source, str(output / f'{label}.py'), 'exec'), namespace)
        with safe_open('artifacts/upstream/model-00001-of-000001.safetensors', framework='pt', device='cpu') as weights:
            actual = namespace['graph'](weights, ref['input_embeddings'], torch.float32)
        if label == 'unchanged_control':
            assert all(torch.equal(v, ref[k]) for k, v in actual.items())
        results[label] = {
            'vs_fp64': oracle.metric(actual['logits'], precise['logits']),
            'vs_native': oracle.metric(actual['logits'], native_logits),
            'first_layer_last_vs_fp64': oracle.metric(actual['layer.0.hidden'][0, -1], precise['layer.0.hidden'][0, -1]),
            'route_set_changes_vs_control': sum(
                (actual[f'layer.{i}.expert_ids'].sort(-1).values != ref[f'layer.{i}.expert_ids'].sort(-1).values).sum().item()
                for i in range(1, 12)),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        }
        print(label, results[label], flush=True)
    results['source_files_sha256'] = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                                     for p in (source_path, Path(__file__))}
    with (folder / 'results.json').open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
