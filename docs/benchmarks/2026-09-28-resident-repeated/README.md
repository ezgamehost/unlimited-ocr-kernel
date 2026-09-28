# Repeated resident CPU comparison

Both engines completed 48 independent requests: two warmups and ten measured
requests for each of four pages. All requests reached EOS, with no failures or
truncations. Rust has lower median page latency on this corpus, but slower
autoregressive decoding; the dense-page p95 is slightly worse. These are sequential
engine runs on a shared M4 Pro, not measurements on an isolated host.

| Page | Rust median / p95 (s) | llama.cpp median / p95 (s) | Median speed ratio | CER, both |
|---|---:|---:|---:|---:|
| english | 11.290 / 16.749 | 14.285 / 26.354 | 1.265× | 0.000000 |
| chinese | 11.511 / 12.363 | 14.213 / 18.980 | 1.235× | 0.000000 |
| table | 11.103 / 14.260 | 14.262 / 17.605 | 1.285× | 0.003984 |
| dense | 15.667 / 20.891 | 16.978 / 20.473 | 1.084× | 0.000000 |

The ratio is llama.cpp median divided by Rust median. Nearest-rank p95 with ten
samples is the maximum; all outliers are retained. CER 0.003984 is the same
one-character table error in both engines. Native BF16 previously avoided that
error; quantization remains an explicit accuracy tradeoff.

| Page | Rust / llama median TTFT (s) | Rust / llama median decode (tokens/s) | Emitted tokens |
|---|---:|---:|---:|
| english | 9.326 / 12.990 | 132.6 / 197.5 | 250 |
| chinese | 9.221 / 12.796 | 135.1 / 203.9 | 289 |
| table | 9.162 / 13.005 | 131.7 / 194.7 | 248 |
| dense | 9.532 / 12.750 | 130.8 / 182.5 | 768 |

Native TTFT uses the internal inference clock; llama TTFT measures first streamed
content/token arrival and includes request transport. These clocks are disclosed
rather than treated as exactly interchangeable. Decode rates use actual decoder
steps, excluding the first token selected from prefill. The native decode clock
covers decoder calls; llama reports server prediction timing. Raw timings remain
available for inspection.

| Process metric | Rust | llama.cpp |
|---|---:|---:|
| Startup wall (s) | 7.164 | 5.041 |
| Peak child RSS (GiB) | 5.989 | 8.079 |

Peak RSS is obtained from `wait4` for each exact child PID over its complete
lifetime, including startup and warmups. Both exited 0 without forced kills.
It is not a per-request memory measurement. Startup is one observation per engine
and is excluded from resident request latency. Rust performs mandatory source and
overlay checksum validation on load; this run does not establish a fresh-process win.

## Matching and differences

- Both runs use eight workers and child environment `VECLIB_MAXIMUM_THREADS=8`,
  `OMP_NUM_THREADS=8`, concurrency one, greedy decoding, and output cap 8192.
- Both use the same ordered request schedule and hashed corpus. Every llama
  request reports `prompt_n=1518` and `cache_n=0`. Native requests explicitly use
  `<image>\ndocument parsing.` to match llama MTMD’s inserted newline.
- All ten measured outputs are identical within each engine for each page.
  Dense-page token IDs and raw text match across engines exactly. English,
  Chinese and table differ by seven, two and one coordinate token IDs respectively;
  emitted token counts and scored text CER match.
- Native uses block32 INT8 decoder weights, dynamic INT8 activation rounding,
  and original BF16-storage vision weights expanded to FP32. llama uses Q8_0
  decoder weights and the metadata-corrected F16 projector with max_tiles=32.
- Native retains all prefix KV plus a 128-generated-token ring and original
  windowed n-gram suppression. This llama build uses full attention. These are
  disclosed semantic differences, not claims of identical internal computation.
- The native and llama phases ran sequentially. Host activity and thermal state
  may vary; these measurements do not establish universal speedups.

## Reproduction and evidence

Run from the repository root:

```sh
.venv/bin/python -m tools.benchmark_resident --engine native \
  --binary target/release/uocr --model artifacts/upstream \
  --int8-weights artifacts/unlimited-ocr-q8-block32.safetensors --int8-activations \
  --output /tmp/uocr-resident-native-new
.venv/bin/python -m tools.benchmark_resident --engine llama \
  --binary /tmp/unlimited-ocr-baseline/bin/llama-b11223/llama-server \
  --model /tmp/unlimited-ocr-baseline/models/model.gguf \
  --mmproj artifacts/mmproj-unlimited-ocr-F16-tiles32.gguf \
  --output /tmp/uocr-resident-llama-new
.venv/bin/python docs/benchmarks/2026-09-28-resident-repeated/summarize.py
```

The final command validates and summarizes the retained adjacent evidence, not
the `/tmp` reruns. New output directories must not already exist. Defaults are
two warmups and ten measured requests per image, seed 42. Each `process.json`
contains the full command, binary and runner SHA256, environment, schedule and
process resources. Individual records retain decoded text/IDs, raw responses,
native requests or llama SSE streams, and completion/accuracy results.

Native runtime source is `e5dd830`; runner source is `ce095b2`. Subsequent commits
before this report changed documentation only. Model and projector hashes are
recorded in [the input manifest](../2026-09-28-thread-controls/input-hashes.json).
See [comparison.json](comparison.json), [native summary](native/summary.json),
and [llama summary](llama/summary.json). No vLLM measurement is claimed: only the
M4 Pro is available. BF16 and fresh-process comparisons remain separate pending work.
