# Powered hybrid Rust vs llama.cpp CPU comparison

All 96 requests completed and passed the raw-record audit: four processes in
Rust/llama/llama/Rust order, each with one warmup and five measured requests per
image. Combined results have ten measurements per image/engine. Every request
reached EOS; all processes exited normally without a forced kill.

| Page | Rust median / p95 (s) | llama median / p95 (s) | llama/Rust median ratio |
|---|---:|---:|---:|
| English | 10.473 / 10.592 | 12.866 / 13.239 | 1.229 |
| Chinese | 10.707 / 10.817 | 13.092 / 13.362 | 1.223 |
| Table | 10.425 / 10.473 | 12.777 / 13.425 | 1.226 |
| Dense | 13.837 / 13.936 | 15.564 / 15.919 | 1.125 |

Nearest-rank p95 is the maximum with ten observations; no outliers were removed.
Rust CER is 0 on all four pages. llama CER is 0 except the table's one-character
error (0.003984). Table output lengths differ: Rust 247 tokens, llama 248. Other
lengths match: English 250, Chinese 289, dense 768. Each engine produced one exact
text/token variant per page across its 12 requests. Equal CER/length does not
imply cross-engine coordinate or token identity.

Rust's median reported decode throughput is 157–158 tokens/s; llama's is 198–207.
Rust wins complete-page latency here, not standalone decode throughput. Reported
decode/TTFT boundaries differ across engines. Native vision still consumes
about 7.1–7.2s, or 52–68% of complete-page latency (`native-stages.json`).

Exact-child lifetime peak RSS was 5.043/5.082GB for the two Rust processes and
10.472/9.726GB for llama (decimal units). These include model loading and all
requests, not a per-page allocation measurement. Startup observations were
Rust 4.770/4.886s and llama 4.822/3.237s; these are only two observations each.

Configuration: shared M4 Pro, AC connected at start and all phase boundaries,
eight workers/eight BLAS threads, concurrency 1, seed 42, cap 8192, CPU-only llama
with cache_n 0, matched newline prompt/prefix 1518. Rust uses packed Q4 body/Q8
head with dynamic INT8 activations and original vision weights expanded to FP32;
llama uses Q8_0 decoder and F16 projector. This compares systems with different
precision and attention/repetition policies, not isolated kernels at equal bits.
These four synthetic selection pages do not establish production OCR quality.

Binary and weight hashes, commands, schedules, thread environment and actual
storage metadata are retained with the raw requests/responses. Rust binary SHA:
`7990f781a918f3f0cfe68fa5d6294f504d56fb6f45524589ef584d6ed948de58`.
A copy is preserved under `artifacts/kernel-baselines/uocr-<SHA>`.

Host snapshots show roughly 10 GB swap in use and ongoing paging/compression.
The 1245.536 s snapshot interval had 952219 pageins, 16923 pageouts, 22661 swapins,
and 0 swapouts. These are host-wide counters, not attributed to either child.
There were no recorded thermal/performance warnings in the snapshots. The
start snapshot was taken after the first process became ready. AC snapshots
do not prove power/thermal isolation. Do not combine these results with the
earlier battery-interrupted series in the parent directory.

Reproduce the audit from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-packed-int4-resident/summarize_repeated.py \
  --input-root docs/benchmarks/2026-09-28-packed-int4-resident/powered-abba-1
```

`repeated-comparison.json` contains all audited distributions and per-phase
resource observations. For a new timed run, use the parent's `run_repeated.py`
with a fresh `--output-root`; existing results are deliberately protected.
