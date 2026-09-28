# llama.cpp CPU smoke baseline — M4 Pro

Executed 2026-09-27 on Apple M4 Pro, 24 GiB RAM, macOS 26.6.2.
This is a logo OCR smoke benchmark, not a document-corpus accuracy or throughput
evaluation. The Rust runtime has not been implemented. vLLM has not been run;
only the local Mac is available.

## Results

Three final measurements, each a fresh process with warm filesystem caches,
10 threads, Q8_0 decoder and F16 vision weights:

| Metric | Run 7 | Run 8 | Run 9 |
|---|---:|---:|---:|
| Process wall time, seconds | 4.469 | 4.574 | 4.545 |
| Vision encoding, milliseconds | 1923 | 2001 | 1937 |
| Decoder evaluation, tokens/second | 251.57 | 242.60 | 240.17 |
| Decoder evaluation steps | 19 | 19 | 19 |
| Peak resident memory, bytes | 8376074240 | 8377204736 | 8377892864 |

Median process wall time: **4.545 seconds**. Median decoder evaluation throughput:
**242.60 tokens/second**, measured over only 19 steps; not a sustained decoding
result. Maximum peak RSS across these runs: **7.803 GiB**. All three exited zero.
Wall time includes loading, CPU weight repacking, image processing, generation
and teardown. Vision and decoder timing fields are runtime-reported; they are
not additive substitutes for wall time. Internal prompt-eval timing includes
multimodal work, so it is not presented as isolated text prefill throughput.

All eight successful runs produced identical bytes:

```text
title [0, 0, 999, 999]<|/det|>Bai du 百度
```

The 430×134 input is upstream `assets/baidu.png`, containing only the Baidu logo.
The output includes an inserted space in “Baidu” and lacks an opening det tag;
this is not evidence of exact structured-output parity. No ground-truth corpus
CER has been measured. Generation stopped below the 128-token cap.

## Reproduce

The exact argv is in `command.json`; artifact origins and hardware are in
`provenance.json`. The artifacts currently reside under
`/tmp/unlimited-ocr-baseline` and are not checked into git. Download the pinned
release/model files from those origins if the temporary directory is removed.

```sh
/usr/bin/time -l /tmp/unlimited-ocr-baseline/bin/llama-b11223/llama-mtmd-cli \
  -m /tmp/unlimited-ocr-baseline/models/model.gguf \
  --mmproj /tmp/unlimited-ocr-baseline/models/mmproj.gguf \
  --image /tmp/unlimited-ocr-baseline/baidu.png \
  -p 'document parsing.' \
  -ngl 0 --device none --no-mmproj-offload --no-op-offload --no-kv-offload \
  -t 10 -tb 10 -c 4096 -n 128 --temp 0 --seed 42 \
  --repeat-penalty 1.0 --no-warmup --jinja -lv 4
```

The CLI inserts its image marker automatically. Logs verify all 13 decoder
layers assigned to CPU, CPU KV buffers, and `CLIP using CPU backend`.
The binary is llama.cpp b11223 / commit
`4da6337767f973e2b4d0797e5b323d77d8565e4a`.

Verified SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| macOS ARM64 release archive | `5bacea12237283699a196194b7f62a0e613438bd5feed694359ed6050492bb3e` |
| Q8_0 language weights | `149574e63ec2681bbfb3d0f83ca1064e041fd6f3a97a62382d3168ad64289848` |
| F16 vision weights | `75e2a133676a82189b10fb1c55b5b32aedc6ede796dd6d8d7c1f901d20ebc7df` |
| Logo input | `3342dd06c42f0abc227adf8e3a3602eda301adfb9af1371f22b9f40c6b5f3339` |

Archive and weight hashes match the publishers' GitHub/Hugging Face metadata.

## Limitations and run history

- Upstream llama.cpp's pinned converter says Unlimited-OCR sliding-window
  metadata is ignored by its decoder, which uses full attention. This baseline
  cannot validate the original model's long-generation attention behavior.
- Q8_0/F16 differs from original BF16 weights. No reference parity is claimed.
- Repetition suppression is disabled here; the official model uses a custom
  n-gram processor. This short output does not exercise that difference.
- A tokenizer EOS/EOG warning is present. Detailed logs list token 1 as EOG,
  and all successful runs stopped below the token limit. Long-output stopping
  still needs separate validation.
- Run 1 failed before inference without `--jinja`; retained for diagnosis.
- Run 2 was the first successful launch: 12.36 seconds, RSS 6440632320 bytes.
  This is not a controlled cold-cache benchmark because earlier loading and
  checksum reads had already touched the files.
- Run 3 enabled debug logging to verify device assignment and timings.
- Runs 4–6 used info logging, which omitted decoder timing details.
- Runs 7–9 used trace logging to collect the final table. No process warmup was
  requested, but OS caches were warm from earlier runs. Timing changed strongly
  during initial runs; this small sample does not characterize that variability.

Next measurement: a versioned document corpus, longer outputs, controlled
warmup/repetitions and accuracy scoring. Compare future Rust CPU runs on the
same inputs and settings, while reporting attention and precision differences.

Sources:
- https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/conversion/deepseek.py
- https://huggingface.co/vimalnakrani/unlimited-ocr-gguf/tree/45cd66ec6b46a7c4de49f376084ecec2b8d3c59a
- https://huggingface.co/baidu/Unlimited-OCR/tree/07dea832e22aefee32ad281d4b80551282e1c168
