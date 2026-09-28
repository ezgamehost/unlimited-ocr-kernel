# Concurrent single-token Q/K/V probe

Stage timing attributes about 19% of fixed-work decode to input normalization and Q/K/V projections. This candidate executes the three independent projections concurrently in the existing pool, with one Rust worker per projection. It changes scheduling only, preserving per-row arithmetic and projection output order.

One/two-worker contexts, multi-token inputs and insufficient aggregate scratch retain the original sequential projection path. The conservative aggregate reservation includes a full weight expansion plus activation/output allowance for all three tasks, so the optimization cannot create a stricter small-budget failure mode.

Validation starts with exact original-weight projection comparisons across worker counts, one/two input rows and a 32 KiB fallback. The full fixed-work ABBA compares every Q8 logit to the preserved accepted shared-expert runtime. Complete OCR and reference gates are required if the candidate has a useful timing gain.

## Controlled AC comparison

Baseline is the accepted shared-expert runtime (`21da757`); intervening commits
contain profiling evidence only. The existing fixed-work Q8 harness runs
1,517 prefix tokens and 512 teacher-forced decode steps, on eight Rust workers
with BLAS/OMP caps of eight. Each of four sequential ABBA processes has one
warmup and three measured repetitions. All processes exit normally without
forced kills; all sixteen full-logit hashes match.

| Metric | Accepted baseline | Concurrent Q/K/V |
| --- | ---: | ---: |
| Median decode including full-logit hashing (s) | 3.800561813 | 3.485923126 |
| Measured range (s), six samples each | 3.760249958–3.849724542 | 3.449515375–3.546544750 |
| Median prefill (s) | 1.521042792 | 1.521310354 |

Decode median is 8.28% lower, with nonoverlapping measured ranges. Prefill is
unchanged within measurement variation. This is an additional native decoder
improvement over the shared-expert baseline, not a new whole-page or llama.cpp
comparison. It must not be multiplied into the published page timings. AC and
normal power-mode snapshots, all samples, identities, source patch and child
resources are under `powered-abba-1/`. The host is shared, not isolated. Process
lifetimes include loading and fixture setup and do not isolate startup cost.

## Validation

The original-weight Q/K/V check passes bitwise comparisons for one/two rows,
one/two/three/four/eight workers and the 32 KiB single-row fallback. The standard
release suite passes 67 tests with 13 opt-in tests ignored. Formatting, Clippy,
original expert/small-scratch checks and the 300-step decoder layer references
pass. The previously disclosed full-model prefill-logit parity gate remains
open; these checks do not resolve it. Independent review has no blocking
findings and retains a minor coverage limitation in `review.md`.

Complete OCR acceptance under `quality/` passes twelve Q8 and twelve hybrid
cases, preserving all raw text (including coordinates), token IDs, EOS,
image-token counts and precision. Both processes exit normally without forced
kills. Existing character errors and the separately documented Q4 32-crop
regression are not fixed by this scheduling change. The candidate is accepted.
Accepted CLI:
`1f1902e4db8a0c9ad5ad79242a439c589ba423ab8462cf52754e8af285c1f193`.
