# Updated Q8 whole-page comparison

The completed 96-request CPU comparison measures a **1.39–1.59× median whole-page
speedup over llama.cpp Q8** on four synthetic pages. All requests reach EOS and
all four processes exit normally without forced kills. Both engines have the
same character error rates and output lengths. All 48 native outputs exactly
match the earlier accepted Q8 benchmark, including coordinate markup and token IDs.

| Page | Native median / p95 (s) | llama median / p95 (s) | Median speed ratio |
| --- | ---: | ---: | ---: |
| English | 8.857 / 9.888 | 14.025 / 15.620 | 1.583× |
| Chinese | 9.019 / 9.294 | 14.302 / 16.375 | 1.586× |
| Table | 8.762 / 9.164 | 13.708 / 15.690 | 1.564× |
| Dense | 12.109 / 12.546 | 16.843 / 19.273 | 1.391× |

Each engine has two warmups and ten measured observations per page, collected
in native/llama/llama/native process order. The nearest-rank p95 is the maximum
of ten samples; no outliers were removed. Both engines produce one output
variant per page. CER is zero except the existing one-character table error
(0.00398406374501992) for both. Output lengths are 250, 289, 248 and 768 tokens.
These synthetic pages have been used during optimization; this does not
establish general document accuracy or performance.

## Decode, startup and memory

Native median decode throughput is **160.2–161.7 tokens/s**, still below llama's
**194.2–205.5 tokens/s**. Native median first-token timings are 7.20–7.26 seconds,
versus 12.52–12.87 seconds for llama. The engines' first-token/decode clock
boundaries differ; raw timings are retained. Whole-page timings measure client
request-to-complete latency and exclude process startup.

Peak child-process RSS is 6.286/5.997 GB for native, versus 7.975/8.804 GB for
llama (decimal GB, process order). Startup observations are 7.367/5.723 seconds
for native and 4.994/4.389 seconds for llama. These are two observations per
engine, not a repeated fresh-process benchmark. Native has lower observed
memory use but higher startup latency in this series.

## Configuration and host conditions

Both engines run CPU-only on the shared M4 Pro with eight Rust/server workers,
eight BLAS/OMP threads, concurrency one, an 8192-token output cap and independent
request caches. The matched newline prompt has 1518 prefix tokens; llama reports
zero prompt-cache reuse on every request. Native uses block32 Q8 weights and
dynamic INT8 decode activations; llama uses Q8_0. Vision storage, attention-cache
and repetition policies differ, so this is a system comparison, not identical
arithmetic or a universal CPU-kernel ratio.

AC power is checked before each phase and after the series; before/after power
profiles show normal mode. The recorded host snapshots contain no thermal or
performance warnings. Swap use falls from 9799.56 to 9600.88 MiB. Whole-host
counters rise by 2,388,818 page-ins and 16,118 swap-ins (16 KiB pages) between
23:23:01 and 23:43:44 UTC on 2026-09-28. These counts span identity hashing,
startup and inference and cannot be assigned exclusively to the benchmark.
Child major faults are native 45,111/689 and llama 193,094/109,946. The host is
not isolated and this is not a no-paging result. Llama's second phase is faster
on several pages; both phases and the slower native English observation remain
in the report. The midpoint snapshot is retained without excluding samples.

## Evidence and scope

The runtime source is `a3a53e13cd23456f8fcedd36319a023e867b605a`, including accepted
shared-expert allocation and concurrent Q/K/V scheduling. The CLI SHA-256 is
`1f1902e4db8a0c9ad5ad79242a439c589ba423ab8462cf52754e8af285c1f193`.
Pinned native overlay, llama executable/model and projector identities are in
`powered-abba-1/repeated-identity.json`; the artifacts and protocol match the
[previous Q8 comparison](../2026-09-28-current-q8/README.md).

`run.py` invokes the existing identity/quality-gated benchmark, captures host
snapshots, and audits all four phases. `audit.log` and `status.json` record
successful completion. `check_native_outputs.py` separately verifies all 48
native responses against the prior accepted outputs; its result is
`native-output-equivalence.json`. The prerequisite twelve Q8 and twelve hybrid
correctness cases, standard tests and decoder references are recorded in the
[concurrent Q/K/V report](../2026-09-28-concurrent-qkv/README.md).

This comparison does not isolate each scheduling change from the previous
full-page run. Their separate controlled native measurements remain separate
from these system-level results. It also does not close the known BF16
prefill-logit tolerance gate, hybrid Q4 stress-case accuracy regression,
remaining precision/fresh-process comparisons, or hardware-dependent vLLM gap.

To reproduce, invoke the existing `run_repeated.py` with `--native-precision int8`,
`--quality-proof docs/benchmarks/2026-09-28-concurrent-qkv/quality/q8`, and a fresh
`--output-root`. Then run `summarize_repeated.py --input-root` on that directory.
Preserve the accepted CLI and avoid concurrent builds/inference while measuring.
