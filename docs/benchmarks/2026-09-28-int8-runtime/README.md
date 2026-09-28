# Native INT8 runtime pilot

Explicit opt-in INT8 decoder inference, CPU only, eight workers on M4 Pro.
The original BF16 vision path and upstream 12-crop/1513-image-token preparation
are retained. Routing, normalization and embedding weights remain original.
The format and artifact checksums are documented in `../2026-09-28-int8-kernels`.

These are exploratory single runs, not the required final repeated benchmark.
The JSONL corpus runs retain one model across English, Chinese, table and dense,
in that order. Each request gets an independent KV cache and a 2048-token cap.
Startup is separately recorded in stderr; resident inference excludes startup.
Raw outputs, token IDs, finish reasons and timings accompany this report.

| Page | Weight-only inference (s) | Dynamic activation inference (s) | Tokens | CER, both modes |
|---|---:|---:|---:|---:|
| English | 18.326 | 18.528 | 250 | 0 |
| Chinese | 17.912 | 18.248 | 289 | 0 |
| Table | 17.351 | 17.633 | 248 | 0.003984 |
| Dense | 37.049 | 44.238 | 1649 | 0 |

All requests finished at EOS. The dynamic activation experiment did not improve
whole-model latency here despite its earlier vocabulary-projection microbenchmark
gain. Its startup was 17.646 seconds. Dense decode took 26.741 seconds versus
22.508 seconds for weight-only. Per-stage values are in `summary.json`.
Equal token counts and normalized CER do not imply identical output: only the
table has identical token IDs between the two precision modes.

```sh
/usr/bin/time -l target/release/uocr --jsonl --threads 8 \
  --int8-weights artifacts/unlimited-ocr-q8-block32.safetensors \
  < docs/benchmarks/2026-09-28-int8-runtime/int8-corpus-requests.jsonl
```

Append `--int8-activations` for the separate experimental W8A8 mode. That mode
rounds single-row projection inputs, including single-row prefill expert groups.
It does not change multirow activation precision.

The checksum library's ARM hardware feature was initially disabled. The first
fresh-process English run loaded in 37.033 seconds and finished in 55.90 seconds.
With that feature enabled, the next run loaded in 24.583 seconds and finished in
43.22 seconds, with identical token IDs and zero normalized CER. Later resident
corpus startup was 16.827 seconds. Filesystem cache and memory pressure were not
controlled, so these samples do not isolate a repeatable hardware-SHA speedup.

`bench_load` separately measured source mapping at 0.002 seconds, source hashing
at 9.483 seconds, overlay loading (including both hashes again) at 18.869 seconds,
and vision expansion at 0.357 seconds. This diagnostic repeats source hashing;
its total must not be treated as production model-load time. Its large gap
between wall and CPU time supports investigating file-backed memory reads.

Weight-only results show zero normalized character errors for English, Chinese
and dense, but one table error (CER 0.003984). The BF16 native pilot had no table
error, so weight quantization introduces a quality regression on this corpus.
All four finish at EOS; dense still emits 1649 tokens. The corrected llama.cpp
pilot emits 768 tokens for dense with a different layout granularity. Do not
interpret total page time as a fixed-work decoder comparison.

The earlier llama.cpp results use fresh processes and different quantization
and attention policies. They cannot be directly ranked against these resident
results. Startup remains substantial and must count in a fresh-process result.
BF16 remains the default; no overall benchmark victory is established.
