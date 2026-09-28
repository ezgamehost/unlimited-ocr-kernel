# Receipt failure traced to prompt formatting

The frozen receipt is recognized correctly by the unchanged native runtime with
its original default prompt, `<image>document parsing.`. Adding the newline used
for the llama-matched benchmark causes immediate EOS in both native precisions
and llama Q8. The image and transcript were not changed.

## Original upstream first-token evidence

`reference_first_token.py` uses the pinned upstream preprocessing, SAM/CLIP,
projector, all twelve decoder layers, output head and original n-gram processor.
It evaluates the same image embeddings with the two prompt forms. Source hashes
and the complete checkpoint SHA256 are verified. Execution is on CPU with
checkpoint weights converted to the specified compute dtype; component forwards
are called directly and decoder weights are expanded one layer at a time.

| Compute dtype | Original prompt, 277 tokens | Newline prompt, 278 tokens |
| --- | --- | --- |
| FP32 | `<|det|>` (128818), score 35.297318 | EOS (1), score 16.709562 |
| BF16 | `<|det|>` (128818), score 35.25 | EOS (1), score 16.625 |

Both original-model CPU calculations reproduce the first-token distinction.
The complete top-ten logits, exact prompts/token IDs, software versions and
fixture hashes are retained in `upstream-*-results.json`, alongside execution
logs. Large tensor fixtures remain under `artifacts/reference-receipt-first-*`.
This is first-token evidence, not full upstream autoregressive generation or a
CUDA-equivalence claim.

## Complete native default-prompt runs

Both BF16 and dynamic INT8 runs emitted 272 tokens, reached EOS, and achieved
zero character error against the frozen receipt transcript. Each exited 0
without a forced kill. The binary hash matches the held-out pilot and repeated
runs: no runtime or preprocessing code was changed to obtain this result.
Raw JSON and exact commands are under `native-original-bf16/` and
`native-original-dynamic-int8/`.

These two checks are fresh-process diagnostic observations, not repeated
performance estimates. BF16 took 7.635470 seconds including startup; dynamic
INT8 took 11.549210 seconds including its checksum/load overhead. The earlier
matched-newline measurements remain unchanged, including their empty outputs
and CER 1.0. They establish the behavior of that benchmark protocol, not failure
of the native product's original default prompt on this receipt.

The native default stays unchanged. Reports must distinguish the original
model recipe from the newline form needed to match this llama port's inserted
newline. Matching prefix tokens can change model behavior and is not always a
quality-neutral adjustment. Do not replace or rewrite prior measurements.

Reproduce from the repository root, using new output directories:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-receipt-diagnostic/reference_first_token.py --output artifacts/reference-receipt-first-f32
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-receipt-diagnostic/reference_first_token.py --dtype bfloat16 --output artifacts/reference-receipt-first-bf16
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-receipt-diagnostic/native_default.py
```
