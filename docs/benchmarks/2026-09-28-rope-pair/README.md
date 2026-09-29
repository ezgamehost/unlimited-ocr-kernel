# Reuse RoPE coefficients across query/key heads

The accepted change computes each rotary coefficient once per query/key row and
channel pair, then reuses it across attention heads. Unlimited-OCR has ten heads,
so this replaces twenty identical coefficient calculations with one. The power,
sine/cosine, multiply/add formulas and absolute positions remain unchanged. It
adds no allocation, model storage, approximation, or precision change.

## Controlled decoder result

| Stage | Baseline median (s) | Candidate median (s) | Reduction |
| --- | ---: | ---: | ---: |
| 512 teacher-forced decode steps, including full-logit hash | 3.315430 | 3.256275 | 1.78% |
| 1517-token prefill | 1.483536 | 1.364281 | 8.04% |

Measured decode ranges are 3.297966–3.332740 seconds versus
3.250759–3.265483; prefill ranges are 1.469264–1.493257 versus
1.358959–1.382702. Both pairs are nonoverlapping. All sixteen repetitions,
including warmups, have identical complete-logit SHA-256:
`3e4fa2e2d048750fba24ad7cde61ccc7c92e1bc2c7436e0d387d867387c6c7da`.

The AC-powered shared M4 Pro runs baseline/candidate/candidate/baseline, with
one warmup and three measured repetitions per process (six measured per variant),
eight Rust workers and eight BLAS/OMP threads. Power and normal-mode settings
are checked before/after each phase. All four processes exit normally without
forced kills. All measured samples are retained. These are resident native
stage measurements, not an updated whole-page comparison against llama.cpp,
a startup measurement, or isolated-host proof.

## Correctness and review

Two new tests first fail for the missing paired API (`red.log`), then pass
(`green.log`). The bitwise test compares against the unchanged original `rope`
implementation across multiple head counts/dimensions, bases, signed zero and
positions through `usize::MAX`. Invalid dimensions, unequal query/key storage
and invalid bases are rejected before either slice changes.

The standard release suite passes **69 tests**, with thirteen opt-in tests
ignored. Formatting and strict all-target clippy pass. Original-weight expert
scheduling and decoder-layer references are explicitly run and pass, including
the fixture's 300 teacher-forced positions. The independent source reviewer
reports no findings (`review.md`).

All twelve Q8 and twelve hybrid Q4/Q8 OCR cases exactly match the accepted
outputs: raw text including coordinate markup, token IDs, EOS, image-token
counts and precision labels. Both quality processes exit normally without forced
kills. This proves compatibility on those cases, not general accuracy. The
known BF16 full-prefill logit tolerance gate and Q4 32-crop stress regression
remain open and are not waived by these results.

## Identities and reproduction

The preserved baseline unit binary is
`7b2821ad84b0991a343790e531252961c7e0c5ce1b6832d5cda8493a90ebbfa9`,
using accepted runtime source through `c435f58`. Candidate unit binary:
`1dfdbb29cbe302527f5701ff458fe0fef1bfa71cf8e8ecca9cd65a30d7b14f90`.
The new accepted CLI is
`c4e2e8ae9c73e0a610bd04c6ebb44b501b0a917ef4fb42c368b3387e5cbde898`,
verified and preserved by the quality harness. Source hashes, overlay identity,
commands and raw process results are retained with each gate.

`run.py --output-name FRESH` repeats the fixed-work ABBA comparison using the
preserved binaries and pinned Q8 overlay. Run from the repository root with
`PYTHONPATH=.` and AC power; avoid concurrent builds or inference while timing.
`validate.py` runs correctness/build gates into a fresh `validation` directory.
The shared `2026-09-28-final-review/quality.py` runner accepts a fresh
`--output-root` and verifies both quantization variants. Reproductions must
record their actual source/executable identities rather than assuming a rebuilt
binary is identical.
