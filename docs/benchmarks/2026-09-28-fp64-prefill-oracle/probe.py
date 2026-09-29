"""Independent, prefill-only eager graph; validate FP32 before evaluating FP64.

FP64 deliberately widens RMS, routing, softmax, and RoPE as well as GEMMs.
This is a mathematical precision experiment, not an upstream FP64 execution mode.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@torch.inference_mode()
def graph(weights, inputs, dtype):
    x = inputs.to(dtype)
    assert x.shape == (1, 277, 1280)
    rows = x.shape[1]
    output = {}

    def weight(name):
        return weights.get_tensor(name).to(dtype)

    def linear(value, name):
        return torch.nn.functional.linear(value, weight(name + '.weight'))

    def norm(value, name):
        return weight(name + '.weight') * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))

    def mlp(value, name):
        return linear(torch.nn.functional.silu(linear(value, name + '.gate_proj'))
                      * linear(value, name + '.up_proj'), name + '.down_proj')

    inv = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=dtype) / 128))
    freq = (inv[None, :, None] @ torch.arange(rows, dtype=dtype)[None, None, :]).transpose(1, 2)
    angles = torch.cat((freq, freq), -1)
    cosine, sine = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    mask = torch.triu(torch.full((1, 1, rows, rows), -torch.inf, dtype=dtype), diagonal=1)

    def rotate(value):
        half = torch.cat((-value[..., 64:], value[..., :64]), dim=-1)
        return value * cosine + half * sine

    for layer in range(12):
        prefix = f'model.layers.{layer}'
        h = norm(x, prefix + '.input_layernorm')
        q, k, v = [linear(h, prefix + '.self_attn.' + p + '_proj')
                   .view(1, rows, 10, 128).transpose(1, 2) for p in ('q', 'k', 'v')]
        q, k = rotate(q), rotate(k)
        attention = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(128) + mask, dim=-1)
        attended = (attention @ v).transpose(1, 2).contiguous().reshape(1, rows, 1280)
        x = x + linear(attended, prefix + '.self_attn.o_proj')
        h = norm(x, prefix + '.post_attention_layernorm')
        name = prefix + '.mlp'
        if layer == 0:
            update = mlp(h, name)
        else:
            flat = h.view(-1, 1280)
            scores = linear(flat, name + '.gate').softmax(dim=-1)
            route_weights, ids = torch.topk(scores, 6, dim=-1, sorted=False)
            output[f'layer.{layer}.expert_ids'] = ids
            output[f'layer.{layer}.expert_weights'] = route_weights
            # Preserve upstream expert grouping and sum order in the FP32 validation.
            order = ids.view(-1).argsort()
            sorted_inputs = flat[order // 6]
            counts = torch.bincount(ids.flatten(), minlength=64).tolist()
            chunks, offset = [], 0
            for expert, count in enumerate(counts):
                if count:
                    chunks.append(mlp(sorted_inputs[offset:offset + count], name + f'.experts.{expert}'))
                    offset += count
            expert_outputs = torch.cat(chunks)
            restored = torch.empty_like(expert_outputs)
            restored[order] = expert_outputs
            update = (restored.view(rows, 6, 1280) * route_weights.unsqueeze(-1)).sum(dim=1)
            update = update.view_as(h) + mlp(h, name + '.shared_experts')
        x = x + update
        assert x.dtype == dtype and torch.isfinite(x).all()
        output[f'layer.{layer}.hidden'] = x.clone()
        print(f'{dtype} layer {layer}', flush=True)
    output['logits'] = linear(norm(x[:, -1:], 'model.norm')[0], 'lm_head')
    return {k: v.contiguous() for k, v in output.items()}


def metric(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.double() - expected.double()).abs()
    return {'max_abs': error.max().item(), 'rms': error.square().mean().sqrt().item(),
            'outside': (error > 1e-4 + 1e-4 * expected.double().abs()).sum().item()}


def main():
    torch.set_num_threads(10)
    destination = Path('artifacts/fp64-prefill-oracle')
    destination.mkdir(exist_ok=False)
    reference_path = Path('artifacts/reference-thread-sensitivity/prefill-threads-10.safetensors')
    reference = load_file(reference_path)
    model = Path('artifacts/upstream/model-00001-of-000001.safetensors')
    assert hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() == '2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6'
    with safe_open(model, framework='pt', device='cpu') as weights:
        fp32 = graph(weights, reference['input_embeddings'], torch.float32)
        # Exact equality is intentional: a failed validation stops before FP64.
        for name, actual in fp32.items():
            assert torch.equal(actual, reference[name]), (name, metric(actual, reference[name]))
        print('FP32 graph exactly matches every upstream hidden state, route and logit', flush=True)
        fp64 = graph(weights, reference['input_embeddings'], torch.float64)
    save_file(fp64, destination / 'prefill.safetensors')
    native_path = Path('artifacts/reference-thread-sensitivity/native-prefill.json')
    native = json.loads(native_path.read_text())
    results = {'fp32_validation': 'bitwise exact, all exported tensors',
               'reference_vs_fp64': metric(reference['logits'], fp64['logits']),
               'native_vs_fp64': metric(torch.tensor(native['logits'])[None], fp64['logits']),
               'layers': {str(i): {
                   'reference': metric(reference[f'layer.{i}.hidden'][0, -1], fp64[f'layer.{i}.hidden'][0, -1]),
                   'native': metric(torch.tensor(native[f'layer.{i}.last_hidden']), fp64[f'layer.{i}.hidden'][0, -1])
               } for i in range(12)},
               'routes_changed': {str(i): (reference[f'layer.{i}.expert_ids'].sort(-1).values != fp64[f'layer.{i}.expert_ids'].sort(-1).values).sum().item() for i in range(1, 12)},
               'sha256': {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                          for p in (reference_path, native_path, model, Path(__file__))}}
    with Path(__file__).with_name('results.json').open('x') as output:
        json.dump(results, output, indent=2)
        output.write('\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
