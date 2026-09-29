# Exact NEON block quantization

The accepted AArch64 kernel quantizes complete 32-value blocks with NEON while
preserving the scalar algorithm: maximum absolute value, FP32 division by 127,
per-value division by that scale, ties-to-even rounding, clamp and signed INT8
conversion. Nonfinite inputs are rejected before quantization; nonzero scale
underflow retains the existing error. Partial blocks and non-AArch64 builds
retain the scalar implementation. No model storage or approximation is added.

## Controlled decoder results

Two independent resident series use the accepted paired-RoPE runtime as baseline.
Each process has one warmup and three measured repetitions; each series has six
measured observations per variant. All measured samples are retained.

| Series | Decode median baseline → candidate (s) | Reduction | Prefill median baseline → candidate (s) |
| --- | ---: | ---: | ---: |
| Baseline/candidate/candidate/baseline | 3.310922 → 3.217794 | 2.81% | 1.391314 → 1.365213 |
| Candidate/baseline/baseline/candidate | 3.261420 → 3.209205 | 1.60% | 1.360801 → 1.366722 |

The first series is noisy, with retained measured outliers of 3.563073 seconds
for baseline and 3.543062 for candidate. The reverse-order confirmation has
nonoverlapping decode ranges: baseline 3.253709–3.282570 seconds and candidate
3.198594–3.221362. Its prefill ranges overlap (1.353728–1.381730 versus
1.350672–1.378377), so no prefill improvement or regression is established.
The conservative confirmed decode improvement is **1.60%**.

All 32 repetitions have identical complete-logit SHA-256:
`3e4fa2e2d048750fba24ad7cde61ccc7c92e1bc2c7436e0d387d867387c6c7da`.
Every sample uses 1517 prefix tokens, 512 teacher-forced decode steps, eight Rust
workers and eight BLAS/OMP threads. Decode timings include hashing all logits.
AC power and normal power mode are checked around each phase. All eight
processes exit normally without forced kills. The M4 Pro is shared, not isolated.
These are native resident stage comparisons, not an updated llama.cpp full-page
result or startup measurement.

## Correctness, lint and review

The missing helper fails first (`red.log`); implementation and expanded tests
then pass. Exact code/scale tests cover varied magnitudes, signed zero, unaligned
input, rounding ties and neighboring floats, subnormal underflow, and maximum
finite values. Integration tests compare scalar-derived quantized products
across complete blocks and row tails and retain malformed-input rejection.

The final standard release suite passes **71 tests**, with thirteen opt-in tests
ignored. Original-weight expert scheduling and the decoder-layer fixture with
300 teacher-forced positions are explicitly run and pass. Formatting and strict
all-target clippy pass. Independent read-only review reports no findings.

The first validation run stops at a clippy rule requiring `as_chunks::<4>().0`
in place of `chunks_exact(4)`; its logs are preserved in `validation-initial`.
After that change, the complete rebuilt unit executable is byte-for-byte equal
to the timed candidate (`post-lint-binary-equivalence.json`), so the timing
measurements still apply. Final validation is separately retained in `validation`.

All twelve Q8 and twelve hybrid Q4/Q8 OCR cases exactly preserve raw text and
coordinates, token IDs, EOS, image-token counts and precision labels. Both
quality processes exit normally without forced kills. This does not close the
known BF16 full-prefill logit tolerance issue or Q4 32-crop stress accuracy failure.

## Identities and reproduction

Baseline source is `d2821ca` (paired RoPE), with preserved unit binary SHA-256
`1dfdbb29cbe302527f5701ff458fe0fef1bfa71cf8e8ecca9cd65a30d7b14f90`.
The candidate unit executable is
`e95db99e865882c9fadf11b87a0d65925f2c80dc4b6dcc1dc02a7733796f0788`.
The accepted CLI, preserved by the quality harness, is
`05616a6d17bae60ceda0fefa365dcdb0f93b6987e00246cd8bb8979a7178ee5e`.

`run.py --output-name FRESH --order abba|baab` repeats the fixed-work comparison
with the preserved binaries and pinned Q8 overlay. Invoke from the repository
root with `PYTHONPATH=.` on AC power and avoid concurrent heavy work. `validate.py`
requires a fresh `validation` output directory. The shared
`2026-09-28-final-review/quality.py --output-root FRESH` performs both OCR gates.
Source/executable hashes, commands, power checks and all raw samples are retained.
