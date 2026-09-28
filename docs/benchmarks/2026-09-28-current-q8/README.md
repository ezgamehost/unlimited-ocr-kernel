# Current Q8 runtime validation and comparison

The complete AC-powered native/llama/llama/native comparison contains 96
audited requests: two warmups and ten measurements per image per engine.
All requests reach EOS, with all four processes exiting normally and no forced
kills. Both engines have the same character error rates and output lengths
on these four synthetic pages. Each engine produces one output variant per page.

| Page | Native median / p95 (s) | llama median / p95 (s) | Median speed ratio |
| --- | ---: | ---: | ---: |
| English | 9.209 / 9.286 | 14.220 / 15.592 | 1.544× |
| Chinese | 9.466 / 9.527 | 14.448 / 14.954 | 1.526× |
| Table | 9.193 / 9.235 | 13.733 / 15.082 | 1.494× |
| Dense | 12.928 / 13.274 | 16.832 / 17.600 | 1.302× |

The nearest-rank p95 is the maximum of ten observations; no outliers were
removed. CER is zero except one table character (0.003984) for both engines.
Output lengths are respectively 250, 289, 248 and 768 tokens. This is a
system-level comparison: native block32 Q8 with dynamically quantized decode
activations differs from llama Q8_0; vision storage, attention and repetition
policies also differ. Both use the matched newline prompt, 1518 prompt tokens,
eight configured workers/BLAS threads, CPU-only execution, independent request
caches, concurrency one and an 8192 output cap. The corpus is synthetic and
has been used during optimization; these results do not establish general
document accuracy or speed.

Native standalone decode remains slower: median 143.5–144.3 tokens/s versus
192.3–205.1 for llama. Median first-token latency is 7.44–7.51s versus
12.54–13.02s; engine clock boundaries differ and raw timings are retained.
Peak child-process RSS is 6.370/5.879 GB for the two native processes versus
9.018/8.993 GB for llama (decimal GB). Startup observations are 5.712/5.705s
versus 5.150/4.073s; two observations do not constitute a repeated fresh-process
benchmark, and resident page latency excludes startup.

This was a shared M4 Pro on AC power, normal power mode, with no recorded
thermal/performance warnings. Host snapshots bracket the whole run, including
identity hashing and startup. Swap use was 9410.19 MiB before and 9330.19 MiB
after; host page-in counters rose by 2,136,119 pages and swap-ins by 4,986
(16 KiB pages). These host counters cannot be assigned exclusively to inference.
Child major faults were 765/689 for native and 193272/70185 for llama. Llama's
second phase was faster than its first; both remain in the report. The result
is not an isolated-host or no-paging claim, nor a measured incremental gain
from the earlier hybrid benchmark, which used different weights and conditions.

Raw requests, child resources, identities, per-phase power checks and the
audited summary are under `powered-abba-1/`; `audit-q8.log` records the audit.
The benchmark source revision is `2e93b0235363a1b61589e5f6eb656bead38dba67`;
the auditor's precision-dependent descriptive note was corrected after the
run and the complete audit rerun. Timing records were not changed.

## Exact-output prerequisite

The current native CLI includes the accepted GELU and head-major cache changes.
Before timing Q8 against llama.cpp Q8, twelve complete OCR cases were run with
the preserved pre-GELU/cache binary and the current binary, using the same Q8
overlay. All text, token IDs, EOS and image token counts match exactly. Both
processes exit normally. The table retains its existing one-character error
(CER 0.003984); the other ten prompt/page cases have zero character errors.

This is preservation of the Q8 baseline, not proof of unrestricted accuracy.
The separately recorded 32-crop case retains the BF16/Q8 transcript and its
0.069117 CER; the hybrid Q4 variant has a much larger regression there. See
the sibling `2026-09-28-hybrid-max32` report. The original model attribution of
the remaining BF16/Q8 stress-case errors remains pending.

The native binary SHA is
`c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57`.
The Q8 overlay SHA is
`b6afd272e6f19de970f97245c83b09e110a20acc8033f2853d0e6739e5fde441`.
`quality.py`, `baseline/` and `quality/` preserve the exact procedure and raw
proof. The runner rejects a hybrid proof when Q8 is selected before creating
an output directory or running inference (`precision-guard.json`). The updated
auditor also successfully rechecked the preceding 96-request hybrid series.

Reproduce from the repository root, selecting a fresh output directory
(the runner refuses an existing directory):

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-packed-int4-resident/run_repeated.py \
  --native-precision int8 \
  --quality-proof docs/benchmarks/2026-09-28-current-q8/quality \
  --output-root docs/benchmarks/2026-09-28-current-q8/powered-abba-reproduction
```

Only after all four phases finish, audit with `summarize_repeated.py` from the
same runner directory and `--input-root` pointing to that fresh series. There
should be no full-page performance claim for a new run until its complete raw
audit passes.
