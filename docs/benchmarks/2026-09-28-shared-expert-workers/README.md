# Shared expert worker allocation probe

The model runs six selected routed experts (width 896) and one shared expert (width 1792) concurrently. The accepted runtime gives all seven tasks serial inner projections. Hypothesis: on the eight-worker configuration, assigning two-way inner projection parallelism to the shared expert reduces its critical path. All tasks still share the existing fixed-size pool; no threads or model allocations are added.

The candidate leaves one-worker inner projections for routed experts and assigns the shared expert min(shared expert count, max(1, pool workers minus routed task count)) workers. The memory partition and low-scratch serial fallback remain unchanged. Every output row and expert reduction retain their original arithmetic order. Multi-token prefill is unchanged.

Validation: focused worker-budget test, identical all-logit checksums through the 512-step fixed-work decoder ABBA, then full suite, reference checks and OCR if timing warrants retention. Battery results remain exploratory until AC confirmation.

## AC fixed-work result

Fresh baseline/candidate/candidate/baseline processes on AC, normal power mode,
eight Rust workers and eight BLAS threads. Each process prefills 1,517 tokens
and runs 512 teacher-forced decode steps for one warmup plus three measured
repetitions. All 16 full-logit SHA-256 values match; all four processes exit
normally without forced kills.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Median decode including full-logit hashing (s) | 3.691855875 | 3.609867500 |
| Measured decode range (s), six samples each | 3.673919167–3.728188333 | 3.577741208–3.635415334 |
| Median prefill (s) | 1.487958042 | 1.483143375 |

Median decode time is 2.22% lower with nonoverlapping measured ranges. This
is a native fixed-work Q8 decoder comparison, not a new whole-page or llama.cpp
result. The host was shared, not isolated. Source and binary identities,
power snapshots, child resource usage and every sample are retained under
`powered-abba-1/`. Prefill source behavior is unchanged; its small timing
difference is not claimed as an optimization. Process lifetimes include loading
and fixture setup and are not isolated startup measurements.

The earlier `battery-abba-1/` is incomplete: power changed to AC during candidate
phase 1 and the guard rejected the series. Its raw output is retained but none
of its timings are included above. The runner now persists child resources
before the after-phase power check, so a future power rejection retains them.

## Validation

The focused worker-budget test covers one, four, eight and ten workers, shared
pool identity and the unchanged scratch partition. The standard release suite
passes 67 tests, with 12 opt-in tests ignored. Formatting and Clippy pass. Both
the original expert scheduling/small-scratch reference and the original
prefill/300-step layer reference pass. The pre-existing full-model logit-parity
failure is still open; these checks do not resolve it. Independent read-only
review found no issues (`review.md`).

The initial validation attempt failed before tests because the script resolved
the Cargo symlink to the rustup dispatcher, changing its executable name. The
script now preserves the Cargo path; the failed attempt is retained separately
in `validation-cargo-dispatch-error/`.

The complete OCR acceptance gate under `quality/` passes all twelve Q8 and
all twelve hybrid cases, preserving raw text (including coordinates), token IDs,
EOS, image-token counts and precision. Both processes exit normally without
forced kills. This preserves existing outputs; it does not fix the known Q4
32-crop regression. The change is accepted. The accepted CLI SHA is
`1846f3dd7495e3fb0e675ce38931de1daae4bda967f6ad13bba65e885a2e93b3`.
