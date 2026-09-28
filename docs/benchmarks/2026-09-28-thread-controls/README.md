# Explicit benchmark threading and resident baseline audit

The benchmark harness now sets child `VECLIB_MAXIMUM_THREADS` and
`OMP_NUM_THREADS` before process startup, without changing the parent
environment. The command-line helper defaults to one BLAS thread; callers can
request a larger BLAS limit up to the worker count with `--blas-threads`.
Each result records these two environment settings, never the full environment.
The new test exercises a real child process and verifies parent isolation and
invalid-limit rejection (RED missing helper, then nine benchmark tests GREEN).

The local Apple SDK `usr/share/man/man7/Accelerate.7` documents setting
`VECLIB_MAXIMUM_THREADS` before Accelerate calls because its value may be cached.
The requested limit is recorded configuration, not a claim that every process
thread is a compute worker or that an OS-level CPU quota was applied.

`native-blas-pilot/` compares one English request with limits one, eight and
unset, in that order. Every output token remains identical and reaches EOS.
These single samples include changing cache/startup state, so they do not
establish an optimal limit. The eight-thread limit is used explicitly for the
resident comparison pilots.

## Rejected server interfaces

`chat-marker-diagnostic/` retains the first llama.cpp server run, with prompt
caching disabled and five independent requests (English twice, then Chinese,
table, dense). The default server response hides special tokens, including
OCR detection delimiters, leaving coordinate text in the transcript. Applying
the existing scorer to that representation yields misleading high CER. These
records are interface diagnostics, **not valid OCR quality comparisons**.
The scoring function has not been relaxed to make them pass.

`media-marker-diagnostic/` retains a failed native completion request. The
pinned server overrides the generic MTMD marker with a per-process random
marker, despite the README's generic marker example. Its `/props` response
exposes the actual `media_marker`; the corrected client reads that property.
The server source establishes this behavior in
[get_media_marker](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/server/server-common.cpp#L134)
and [server properties](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/server/server-context.cpp#L4634).

The corrected client uses `/completion`, `--special`, the exact image marker
followed immediately by `document parsing.`, and requests raw token IDs. It
keeps `cache_prompt=false`, disables the server prompt cache, uses one slot,
and disables context shifting. The full output cap is 8,192 tokens.

## Prompt boundary correction

The pinned [MTMD DeepSeek-OCR branch](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/mtmd/mtmd.cpp#L886)
unconditionally appends a newline after image embeddings. Earlier native
pilots used `<image>document parsing.`, giving a 1,517-token prefix, while
llama.cpp used 1,518 tokens. Matching native input explicitly uses
`<image>\ndocument parsing.`. The tokenizer suffix is `[201, 34030, 76466, 16]`,
versus `[34030, 76466, 16]` without the newline. The runtime default is unchanged.

This explains the prior dense-output granularity discrepancy: with the matched
boundary, both engines emit **exactly the same 768 token IDs and raw text**.
Previously the native default emitted 1,649 tokens with a box per line. Reducing
that output count is a prompt correction, not a kernel throughput improvement.
Previous native-vs-native fixed-work kernel measurements used identical inputs
within each comparison and remain valid for that workload.

## Matched resident pilot

Both engines use eight workers, requested BLAS limit eight, cap 8,192, the same
12-crop policy and the aligned 1,518-token prefix. The pinned llama server uses
Q8_0 decoder weights and F16 vision weights; native uses block-32 INT8 decoder
weights/dynamic INT8 activations and original BF16 vision weights. Attention
and n-gram policies retain the differences documented in earlier reports.
These are sequential pilot runs on a non-isolated host, not the required final
two-warmup/ten-repetition benchmark. English is deliberately requested twice.

Special-token output preserves detection delimiters but also emits EOS. The
adapter removes exactly one terminal EOS only when `stop_type=eos`, the last
raw token ID is 1, and the rendered suffix matches `/props.eos_token`. It keeps
raw responses and token arrays intact. It never removes interior literal text
or adjusts the OCR scorer. The initial unadapted CER is retained separately.
A regression test covers matching evidence, wrong stop/token evidence, missing
markers and interior literal EOS text.

| Page / request | Native wall (s) | llama.cpp wall (s) | Emitted tokens, both | Text CER, both |
|---|---:|---:|---:|---:|
| english / 0 | 12.537 | 15.062 | 250 | 0.000000 |
| english / 1 | 12.608 | 15.338 | 250 | 0.000000 |
| chinese / 2 | 12.908 | 15.664 | 289 | 0.000000 |
| table / 3 | 11.512 | 19.541 | 248 | 0.003984 |
| dense / 4 | 14.647 | 20.152 | 768 | 0.000000 |

The non-dense pages differ in a few coordinate token IDs; only the dense page
has complete token identity across engines. Text CER matches on all pages.
All completions reach EOS. Five zero-cache prompt starts and fifteen image
encoding batches in the llama log prove repeated requests re-encode vision;
all response cache counts are zero. Server timing `prompt_ms` includes vision.
Raw predicted-token counts include EOS; emitted-token counts above exclude it.

The pilot's server shutdown signals the process group, which prevents
`/usr/bin/time` from producing its final RSS summary. Do not infer server peak
RSS from that log. Final benchmarking must collect server resource usage with
a direct child process and completed wait/resource accounting.

Validation: Python suite ran 22 tests successfully, with two artifact-dependent
tests skipped. Input corpus hashes, baseline GGUF/projector hashes and executable
hashes are recorded in `input-hashes.json`.

Reproduce either pilot from the repository root, choosing a new output directory:

```sh
PYTHONPATH="$PWD" .venv/bin/python docs/benchmarks/2026-09-28-thread-controls/run_native_pilot.py --output artifacts/replay-native
PYTHONPATH="$PWD" .venv/bin/python docs/benchmarks/2026-09-28-thread-controls/run_llama_pilot.py --output artifacts/replay-llama
```

These helpers use the recorded local model/binary paths and perform five
requests, not the final repetition protocol. Run them sequentially. The llama
helper now applies the tested terminal-EOS adapter during collection rather
than as the postprocessing step used on the retained raw pilot.
