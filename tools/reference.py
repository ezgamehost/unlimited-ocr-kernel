"""CPU reference fixtures from pinned upstream code, without CUDA infer helpers."""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch
from PIL import ImageOps
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoTokenizer

from tools.fetch_model import REVISION, sha256


def upstream(directory: Path):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['revision'] != REVISION:
        raise ValueError('unexpected upstream revision')
    for name, entry in manifest['files'].items():
        if name.endswith('.py') and sha256(directory / name) != entry['sha256']:
            raise ValueError(f'modified upstream source: {name}')
    package_name = '_uocr_upstream'
    if package_name in sys.modules and sys.modules[package_name].__path__ != [str(directory)]:
        raise ValueError('cannot mix upstream directories in one process')
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules.setdefault(package_name, package)
    return importlib.import_module(package_name + '.modeling_unlimitedocr')


def prepare_inputs(directory: Path, image_path: Path, mode: str):
    if mode not in ('base', 'gundam'):
        raise ValueError(f'unknown image mode {mode}')
    source = upstream(directory)
    image = source.load_image(str(image_path)).convert('RGB')
    transform = source.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    crops, grid = [], (1, 1)
    if mode == 'gundam' and (image.width > 640 or image.height > 640):
        crops, grid = source.dynamic_preprocess(image)
    global_view = ImageOps.pad(image, (1024, 1024), color=(127, 127, 127))
    # Upstream explicitly rounds inputs to BF16 before calling the model.
    tensors = {'global': transform(global_view).to(torch.bfloat16).float().unsqueeze(0)}
    for i, crop in enumerate(crops):
        tensors[f'crop.{i}'] = transform(crop).to(torch.bfloat16).float().unsqueeze(0)
    image_tokens = 273 + ((10 * grid[0] + 1) * 10 * grid[1] if crops else 0)
    prompt = source.format_messages([
        {'role': '<|User|>', 'content': '<image>document parsing.'},
        {'role': '<|Assistant|>', 'content': ''}], sft_format='plain', system_prompt='')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    before, after = prompt.split('<image>')
    ids = [0] + source.text_encode(tokenizer, before, bos=False) + [128815] * image_tokens + source.text_encode(tokenizer, after, bos=False)
    metadata = {'schema_version': 1, 'revision': REVISION, 'mode': mode,
                'image_sha256': sha256(image_path), 'crop_grid': list(grid),
                'image_tokens': image_tokens, 'token_ids': ids, 'prompt': prompt,
                'input_dtype': 'BF16 rounded, exported as F32', 'reference_device': 'cpu'}
    return tensors, metadata


@torch.inference_mode()
def export_vision(directory: Path, inputs: dict, metadata: dict, dtype=torch.float32):
    """Run original SAM and CLIP forwards, changing only device and compute dtype."""
    source = upstream(directory)
    with torch.device('meta'):
        sam = source.build_sam_vit_b()
        clip = source.build_clip_l()
    weights_path = directory / 'model-00001-of-000001.safetensors'
    with safe_open(weights_path, framework='pt', device='cpu') as weights:
        def state(prefix):
            result = {}
            for key in weights.keys():
                if key.startswith(prefix):
                    value = weights.get_tensor(key)
                    result[key[len(prefix):]] = value.to(dtype) if value.is_floating_point() else value
            return result
        sam.load_state_dict(state('model.sam_model.'), assign=True, strict=True)
        clip_state = state('model.vision_model.')
        # HF omits this deterministic buffer from the checkpoint; recreate the
        # exact arange buffer constructed by CLIPVisionEmbeddings on CPU.
        clip_state.setdefault('embeddings.position_ids', torch.arange(clip.embeddings.num_positions).expand(1, -1))
        clip.load_state_dict(clip_state, assign=True, strict=True)
        projector = weights.get_tensor('model.projector.layers.weight').to(dtype)
        bias = weights.get_tensor('model.projector.layers.bias').to(dtype)
        newline = weights.get_tensor('model.image_newline').to(dtype)
        separator = weights.get_tensor('model.view_seperator').to(dtype)
    sam.eval(); clip.eval()
    outputs = {}

    def encode(name):
        image = inputs[name].to(dtype)
        a = sam(image)
        b = clip(image, a)
        combined = torch.cat((b[:, 1:], a.flatten(2).permute(0, 2, 1)), dim=-1)
        projected = torch.nn.functional.linear(combined, projector, bias)
        outputs[name + '.sam'] = a.float().contiguous()
        outputs[name + '.clip'] = b.float().contiguous()
        outputs[name + '.projected'] = projected.float().contiguous()
        return projected[0]

    def add_newlines(grid):
        rows = newline.expand(grid.shape[0], 1, -1)
        return torch.cat((grid, rows), dim=1).reshape(-1, 1280)

    global_projected = encode('global')
    global_sequence = add_newlines(global_projected.reshape(16, 16, 1280))
    gx, gy = metadata['crop_grid']
    if gx * gy > 1:
        local = torch.stack([encode(f'crop.{i}') for i in range(gx*gy)])
        local = local.reshape(gy, gx, 10, 10, 1280).permute(0, 2, 1, 3, 4).reshape(gy*10, gx*10, 1280)
        sequence = torch.cat((add_newlines(local), global_sequence, separator[None]), dim=0)
    else:
        sequence = torch.cat((global_sequence, separator[None]), dim=0)
    if sequence.shape[0] != metadata['image_tokens']:
        raise ValueError('image embedding count does not match prompt')
    outputs['image_embeddings'] = sequence.float().contiguous()
    return outputs


@torch.inference_mode()
def export_decoder(directory: Path, image_embeddings, metadata: dict, steps=300, dtype=torch.float32):
    """Layerwise original forwards bound weight expansion to one decoder layer.

    Teacher forcing permits evaluation of all positions of a layer before
    loading the next layer; this is mathematically the same autoregressive graph.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    source = upstream(directory)
    decoder = importlib.import_module(source.__package__ + '.modeling_deepseekv2')
    config = source.UnlimitedOCRConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = 'eager'
    config.sliding_window = None
    config._ring_window = 128
    prompt_ids = torch.tensor(metadata['token_ids'], dtype=torch.long)
    decode_ids = (torch.arange(steps, dtype=torch.long) * 997 + 23) % config.vocab_size
    ids = torch.cat((prompt_ids, decode_ids))
    prefix = len(prompt_ids)
    outputs = {'decode_token_ids': decode_ids, 'input_token_ids': ids}
    with safe_open(directory / 'model-00001-of-000001.safetensors', framework='pt', device='cpu') as weights:
        hidden = weights.get_tensor('model.embed_tokens.weight')[ids].to(dtype).unsqueeze(0)
        image_positions = (prompt_ids == 128815).nonzero().flatten()
        if len(image_positions) != len(image_embeddings):
            raise ValueError('image/prompt embedding count mismatch')
        hidden[0, image_positions] = image_embeddings.to(dtype)
        outputs['input_embeddings'] = hidden.float().clone()
        mask = torch.triu(torch.full((1, 1, prefix, prefix), float('-inf'), dtype=dtype), diagonal=1)
        for layer_id in range(config.num_hidden_layers):
            with torch.device('meta'):
                layer = decoder.DeepseekV2DecoderLayer(config, layer_id)
            key_prefix = f'model.layers.{layer_id}.'
            state = {key[len(key_prefix):]: weights.get_tensor(key).to(dtype)
                     for key in weights.keys() if key.startswith(key_prefix)}
            layer.load_state_dict(state, assign=True, strict=True)
            del state
            # Rotary frequencies are a nonpersistent buffer initialized on meta.
            layer.self_attn.rotary_emb = LlamaRotaryEmbedding(config).to(dtype)
            layer.eval()
            cache = DynamicCache()
            routed = []
            handle = None
            if hasattr(layer.mlp, 'gate'):
                handle = layer.mlp.gate.register_forward_hook(
                    lambda _m, _args, result: routed.append((result[0].clone(), result[1].float().clone())))
            chunks = [layer(hidden[:, :prefix], attention_mask=mask,
                            position_ids=torch.arange(prefix).unsqueeze(0),
                            past_key_value=cache, use_cache=True)[0]]
            for position in range(prefix, len(ids)):
                chunks.append(layer(hidden[:, position:position+1],
                                    position_ids=torch.tensor([[position]]),
                                    past_key_value=cache, use_cache=True)[0])
            hidden = torch.cat(chunks, dim=1)
            outputs[f'layer.{layer_id}.hidden'] = hidden.float().clone()
            if handle:
                handle.remove()
                outputs[f'layer.{layer_id}.expert_ids'] = torch.cat([x[0] for x in routed])
                outputs[f'layer.{layer_id}.expert_weights'] = torch.cat([x[1] for x in routed])
            print(f'reference decoder layer {layer_id}: {len(ids)} positions', flush=True)
            del layer, cache, chunks
        positions = [prefix-1] + [prefix+i for i in [0, 1, 127, 128, 129, 255, 256, 299] if i < steps]
        selected = hidden[:, positions].float()
        norm = weights.get_tensor('model.norm.weight').float()
        selected = selected * torch.rsqrt(selected.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps)
        selected = (selected * norm).to(dtype)
        # Only a handful of logits rows are needed to test boundaries.
        head = weights.get_tensor('lm_head.weight').to(dtype)
        outputs['logits'] = torch.nn.functional.linear(selected[0], head).float()
        outputs['logits_positions'] = torch.tensor(positions)
    return {key: value.contiguous() for key, value in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/upstream'))
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--mode', choices=['base', 'gundam'], default='base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    torch.set_num_threads(10)
    tensors, metadata = prepare_inputs(args.model, args.image, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, args.output / 'preprocess.safetensors')
    if args.vision or args.decoder:
        dtype = getattr(torch, args.dtype)
        features = export_vision(args.model, tensors, metadata, dtype)
        save_file(features, args.output / 'vision.safetensors')
        metadata['compute_dtype'] = args.dtype
        metadata['adaptations'] = ['CPU execution instead of CUDA', 'model components called directly; original forward math retained', 'BF16 checkpoint weights converted to requested compute dtype']
        if args.decoder:
            decoder_features = export_decoder(args.model, features['image_embeddings'], metadata, args.steps, dtype)
            save_file(decoder_features, args.output / 'decoder.safetensors')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
