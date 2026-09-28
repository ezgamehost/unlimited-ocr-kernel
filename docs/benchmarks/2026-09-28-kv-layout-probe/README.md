# Head-major KV cache candidate

Status: candidate retained for AC confirmation after exact quality gates and
a positive exploratory battery comparison. The earlier accepted NEON GELU
baseline remains separately preserved. No updated llama.cpp claim is made.

## Isolated stride probe

Twelve synthetic layer caches have 1,641 active tokens, ten heads and head
dimension 128, matching a filled primary-page prefix/ring shape. Both physical
layouts hold identical FP32 values. Calls use the existing checked Accelerate
GEMV wrapper, thread-local BLAS guard, scale and softmax, with eight Rayon
workers. Every output bit agrees across layouts for all twelve layers.

One process runs token/head/head/token order, each phase one warmup plus three
measurements. A measurement runs 128 steps over all twelve layers. Median time
is 0.214699833 seconds for token-major and 0.1686191875 for head-major: 21.46%
lower in this isolated workload. Data preparation is outside the timer. This
is not a full-model decode result; it excludes packing and interleaved model
weight traffic. Raw code, timings and identity are in `probe.rs.txt`, `run.log`,
`summary.json` and `process.json`.

## Integrated candidate

The runtime converts each completed prefill-layer cache once. Each head has
fixed prefix+window capacity and an explicit active count. Prefix entries and
physical ring slots retain their order; each new token updates the corresponding
slot in every head. Original token-major construction remains supported. The
conversion retains no permanent duplicate, and allocation failure leaves the
original cache intact. Decoder prefill timing includes conversion cost.

Both attention wrappers share arithmetic and scratch checks. Head-major access
ignores unused capacity, including NaN sentinels in the direct test. Tests cover
one and eight workers, scalar and accelerated backends, minimum scratch fallback,
active counts, partial rings and multiple wraps. The new APIs were absent in the
recorded RED run; the candidate suite passes 62 tests with ten ignored, including
the diagnostic probe. Original-model decoder references explicitly exercise
both layouts across 300 decode steps at layers 0, 1 and 11. The decoder and
vision reference commands pass, as do formatting and Clippy.

The fixed-work candidate checksum matches the established full-logit SHA in
all four repetitions:
`c129d845d9d0f6b2b1f7bc1aaa7741d1a9fb95b428a1016f519df94acb862f3a`.
`checksum.json` records this as a correctness run on battery, not performance
evidence. The separate twelve-case OCR gate completed successfully: text, token
IDs, EOS termination and image token counts match exactly for all cases. The
process exited normally. `quality/` and `candidate-cli.json` retain the proof
and preserved candidate executable identity.

## Outstanding performance gate

`abba/` contains prepared binary identities and the candidate patch, but no
timed phases. The laptop disconnected from AC before phase zero; the driver
stopped at its power guard (`power-guard.json`). This failed attempt is not a
completed comparison and must not be combined with a future series.

Once AC is available, use a fresh output namespace:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-kv-layout-probe/run.py \
  --output-name powered-abba-1 \
  --prepared docs/benchmarks/2026-09-28-kv-layout-probe/abba/identity.json
```

The prepared executables are immutable by recorded SHA; both include the same
accepted model and precision policies. This command must run separately from
builds, profiles and other inference. Acceptance still requires a useful
full-decoder gain. A full-page comparison is a separate remaining requirement.

A separately labeled exploratory battery series is also supported with
`--power-source battery`, requiring at least 50% charge and reported powermode 0.
It checks power source and profile before and after each phase. Such a series
must use its own output namespace and is not combined with AC measurements;
it cannot update the AC-powered llama.cpp performance claim.

## Completed exploratory battery comparison

`battery-abba-1/` completed all four processes normally, with six measured
repetitions per variant after warmups. All sixteen full-logit hashes agree with
the established checksum. Source/profile checks remained on battery before and
after each phase. The snapshot reported no thermal/performance warnings. The
host is shared; individual timings vary and are all retained.

| Metric | Token-major | Head-major |
|---|---:|---:|
| Median 512-step decode including checksum | 3.499262s | 3.356351s |
| Observed decode range | 3.435377–3.614068s | 3.261271–3.406413s |
| Median prefill, including cache conversion | 1.653581s | 1.645228s |

Median decode time is 4.08% lower, with no overlap in the measured ranges.
Prefill does not show a measured regression in this series, but its variation
does not establish that conversion is free. This is a battery fixed-work result,
not a full-page or AC result. AC confirmation and a new full-page comparison
remain pending. The candidate CLI is preserved at the path in
`candidate-cli.json` with SHA
`c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57`.
