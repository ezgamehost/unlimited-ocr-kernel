# Updated full-page comparison after vision optimization

All 96 requests completed and passed the raw-record audit. Rust uses the accepted
SAM attention tiles and parallel vision normalization, with the same packed Q4
body/Q8 output head as the preceding comparison. The unchanged llama.cpp binary,
Q8_0 model and F16 projector were verified by checksum before timing.

| Page | Rust median / p95 (s) | llama median / p95 (s) | llama/Rust median ratio |
|---|---:|---:|---:|
| English | 9.0777 / 9.1214 | 12.9433 / 13.2803 | 1.426 |
| Chinese | 9.3222 / 9.3598 | 13.2578 / 13.7052 | 1.422 |
| Table | 9.0302 / 9.1113 | 12.9774 / 13.2680 | 1.437 |
| Dense | 12.4918 / 12.5611 | 15.6892 / 15.8434 | 1.256 |

Four sequential processes ran Rust/llama/llama/Rust. Each process performed one
warmup and five measured requests per image, yielding ten measurements and two
warmups per image/engine. Nearest-rank p95 is the maximum with ten observations;
all outliers are retained. Every request reached EOS, every process exited
normally, and none required a forced kill. These are resident request times;
loading and initialization are measured separately.

Rust CER is 0 on every page. llama CER is 0 except its one-character table error
(0.003984). Table generation lengths differ: 247 Rust versus 248 llama tokens.
Other lengths match: 250 English, 289 Chinese, 768 dense. Each engine produced
one exact text/token variant per page across its 12 requests. Equal CER and
length do not establish cross-engine coordinate or token identity.

Median reported token-decoding rates remain slower for Rust: 156 tokens/s
versus 198–209 for llama. Engine timing boundaries differ; this is a complete-page
system win, not a decode-kernel win. Rust vision medians are 5.72–5.79s and account
for 46–63% of page latency (`native-stages.json`). Prefill remains about 1.59s.

Exact-child lifetime peak RSS (decimal GB): Rust 5.055/5.057, llama 9.798/9.294.
These are per-process observations covering loading and all requests, not
per-page allocation measurements. Startup observations: Rust 4.895/5.019s,
llama 5.029/3.867s. Two startup observations per engine are not a robust
fresh-process benchmark.

## Configuration and limits

Shared M4 Pro, eight Rust/llama workers, eight BLAS threads, concurrency 1,
seed 42, cap 8192, CPU-only llama with cache_n 0, matched newline prompt and
prefix 1518. Original vision weights are expanded to FP32 in Rust; llama's
projector is F16. Decoder storage, attention and repetition policies differ.
This compares systems with differing precision, not isolated equal-bit kernels.
The four synthetic primary pages were used during development, so they do not
establish quality on unseen production documents.

AC was checked before hashing and each phase. Host snapshots report roughly
9.7GB swap in use and no recorded thermal/performance warnings. Over the
1189.510s snapshot interval, host counters rose 967333 pageins, 24743 pageouts,
2206 swapins and 0 swapouts. Paging/compression counts are host-wide and cannot
be attributed to either child. The first snapshot follows native startup.
These measurements do not claim host isolation or universal hardware results.

Native executable SHA256:
`7baaad44d8ae6f56fd6f470d29271a0b0529be83a8eacd53b69d60c9140c08ba`.
Its source is `3f0aa86`; the quality-proof runner update is `bd6929b`.
The benchmark identity includes hashes of the completed 12-case OCR proof and
requires it to match the actual executable and overlay manifest. Original-model
reference checks and vision ABBA experiments are retained in the linked
[SAM tile](../../2026-09-28-vision-tiles/README.md) and
[parallel normalization](../../2026-09-28-parallel-norm/README.md) reports.

The earlier completed comparison remains in `../powered-abba-1/`. It used the
pre-optimization executable; these are separate experiments, not interchangeable
samples. No requests from the battery-interrupted parent series are included.

Reproduce the audit from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-packed-int4-resident/summarize_repeated.py \
  --input-root docs/benchmarks/2026-09-28-packed-int4-resident/powered-abba-2
```

See the parent README for a new timed run with a fresh output directory.
`repeated-comparison.json` retains the full distributions and per-phase resources;
all raw requests, outputs, decoded text, timing and protocol metadata are present.
