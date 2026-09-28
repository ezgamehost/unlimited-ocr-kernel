"""Original upstream CPU first-token diagnostic for both receipt prompt forms."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from tools.fetch_model import sha256
from tools.reference import prepare_inputs, export_vision, export_decoder, upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    directory, image = Path('artifacts/upstream'), Path('benchmarks/heldout/receipt.png')
    source = upstream(directory)
    inputs, original = prepare_inputs(directory, image, 'gundam')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    dtype = getattr(torch, args.dtype)
    checkpoint_hash = sha256(directory / 'model-00001-of-000001.safetensors')
    assert checkpoint_hash == '2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6'
    assert original['image_tokens'] == 273
    features = export_vision(directory, inputs, original, dtype)
    save_file(features, args.output / 'vision.safetensors')
    results = {}
    for name in ('original', 'matched_newline'):
        metadata = dict(original)
        if name == 'matched_newline':
            metadata['prompt'] = original['prompt'].replace('<image>', '<image>\n', 1)
            before, after = metadata['prompt'].split('<image>')
            metadata['token_ids'] = ([0] + source.text_encode(tokenizer, before, bos=False)
                + [128815] * metadata['image_tokens'] + source.text_encode(tokenizer, after, bos=False))
        assert len(metadata['token_ids']) == (277 if name == 'original' else 278)
        tensors = export_decoder(directory, features['image_embeddings'], metadata, steps=0, dtype=dtype)
        save_file(tensors, args.output / f'{name}-decoder.safetensors')
        scores = tensors['logits'].clone()
        ids = torch.tensor([metadata['token_ids']], dtype=torch.long)
        scores = source.SlidingWindowNoRepeatNgramProcessor(35, 128)(ids, scores)
        values, indices = scores[0].topk(10)
        result = dict(prompt=metadata['prompt'], prefix_tokens=len(metadata['token_ids']),
            greedy_token_id=int(indices[0]), eos_token_id=tokenizer.eos_token_id,
            immediately_eos=int(indices[0]) == tokenizer.eos_token_id,
            top_tokens=[dict(token_id=int(i), score=float(v), decoded=tokenizer.decode([int(i)], skip_special_tokens=False))
                        for i, v in zip(indices, values)],
            metadata=metadata, decoder_sha256=sha256(args.output / f'{name}-decoder.safetensors'))
        results[name] = result
        print(json.dumps(dict(prompt_variant=name, greedy_token_id=result['greedy_token_id'],
                              immediately_eos=result['immediately_eos'])), flush=True)
        del tensors, scores
    report = dict(compute_dtype=args.dtype, checkpoint_sha256=checkpoint_hash,
        torch_version=torch.__version__, transformers_version=importlib.metadata.version('transformers'),
        reference_tool_sha256=sha256(Path('tools/reference.py')), script_sha256=sha256(Path(__file__)),
        adaptations=['Original component forward functions on CPU with checkpoint weights converted to requested dtype.',
                     'Original preprocessing and n-gram processor; only prompt newline varies.',
                     'First-token diagnostic, not full autoregressive generation or a CUDA parity claim.'], results=results)
    (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
