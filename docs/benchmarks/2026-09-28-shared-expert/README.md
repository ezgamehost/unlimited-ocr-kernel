# Schedule shared and routed experts together

Base: `09b05fd` (four-row SDOT). Candidate: `candidate.patch`.

Single-token MoE execution now schedules the shared expert and selected routed
experts in one indexed parallel collection. Each task uses the existing worker
pool with serial inner projections. Accumulation still starts from the shared
expert output and adds routed outputs in the original expert order. Multi-token
execution retains its existing path.

Scratch eligibility conservatively reserves the largest expert width for every
concurrent task, including the shared expert. Insufficient scratch falls back
to serial expert execution. This is intentionally conservative; it does not
promise parallelism at every budget that could theoretically fit.

## Controlled decoder comparison

Two release executables run the existing ignored fixed-work benchmark from the
previous report, with identical fixtures and harness. Sequential process order:
base, candidate, candidate, base. Each process has one warmup and three measured
repetitions. Each repetition prefills a new 1,517-token cache and executes 512
teacher-forced steps cycling the same recorded English tokens. All-logits
checksum computation is included in decode duration. Vision and load are
excluded; this is not an OCR completion or a llama.cpp comparison.

Six measured samples per version give medians of **4.576706 seconds** for the
base and **4.031535 seconds** for the candidate: **11.9% less elapsed time**,
or 1.135x throughput. All sixteen repetitions, including warmups, retain the
same logits checksum, 188943474296580907. Raw logs and every timing are in
`fixed-work.json`. The benchmark driver ran no concurrent builds or inference workloads.
The machine was not isolated: a later full-OCR process snapshot showed
substantial unrelated CPU activity. Timing uncertainty remains, even with
reversed process order; do not treat these samples as final performance claims.

The original-weight decoder tests pass at unchanged tolerance, including 300
positions across cache ring wraps and identical expert results at one/eight
workers and a 32 KiB scratch budget.

`fresh/` records the four-page, eight-worker dynamic INT8 OCR pilot with original
BF16 vision, a 2,048-token cap, full integrity checks and fresh processes. These
pilots check output and overall behavior; final repeated, matching-lifetime
comparisons against llama.cpp remain outstanding.

All four OCR outputs retain exactly the previous token IDs and reach EOS.
CER remains zero for English, Chinese and dense text, with the existing table
one-character error (CER 0.003984). Dense decode is 13.183 seconds versus the
previous 14.873-second pilot; this is a single-page timing observation.

| Page | Load (s) | Vision (s) | Prefill (s) | Decode (s) | Process wall (s) |
|---|---:|---:|---:|---:|---:|
| english | 7.923 | 10.826 | 2.114 | 2.372 | 23.836 |
| chinese | 8.133 | 8.206 | 2.189 | 2.542 | 21.367 |
| table | 6.918 | 7.647 | 1.942 | 1.982 | 18.713 |
| dense | 5.788 | 8.063 | 2.230 | 13.183 | 29.639 |

Verification: full Rust suite 41 passed, eight artifact-dependent tests
ignored by default; two original-weight decoder tests run explicitly and
passed; clippy all targets with warnings denied passed.
