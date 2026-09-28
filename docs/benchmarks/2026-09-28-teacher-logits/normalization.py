"""Compare RMSNorm reduction precision on original exported layer inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(8)
root = Path(__file__).resolve().parent
rows = []
with safe_open('artifacts/reference-logo-f32/decoder.safetensors', framework='pt', device='cpu') as trace, \
     safe_open('artifacts/upstream/model-00001-of-000001.safetensors', framework='pt', device='cpu') as weights:
    for layer in range(12):
        name = 'input_embeddings' if layer == 0 else f'layer.{layer-1}.hidden'
        x = trace.get_tensor(name).float().reshape(-1, 1280)[:277]
        weight = weights.get_tensor(f'model.layers.{layer}.input_layernorm.weight').float()
        upstream = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)) * weight
        data = x.numpy()
        variance = np.zeros(len(data), dtype=np.float32)
        for column in range(data.shape[1]):
            variance = variance + data[:, column] * data[:, column]
        scale = np.float32(1) / np.sqrt(variance / np.float32(1280) + np.float32(1e-6))
        sequential = data * scale[:, None] * weight.numpy()
        exact = (data.astype(np.float64) /
                 np.sqrt(np.mean(data.astype(np.float64)**2, axis=1, keepdims=True) + float(np.float32(1e-6)))) * weight.numpy()
        fp64_variance = np.mean(data.astype(np.float64)**2, axis=1).astype(np.float32)
        reduced64 = data * (np.float32(1) / np.sqrt(fp64_variance + np.float32(1e-6)))[:, None] * weight.numpy()
        stats = {}
        for label, values in [('upstream', upstream.numpy()), ('sequential_fp32', sequential), ('fp64_reduction', reduced64)]:
            stats[label] = dict(max_error_vs_fp64=float(np.max(np.abs(values-exact))),
                                max_difference_vs_upstream=float(np.max(np.abs(values-upstream.numpy()))),
                                last_row_difference_vs_upstream=float(np.max(np.abs(values[-1]-upstream.numpy()[-1]))))
        rows.append(dict(layer=layer, statistics=stats))
(root / 'normalization.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
