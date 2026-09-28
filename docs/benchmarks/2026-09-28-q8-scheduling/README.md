# Finer Q8 scheduling — not retained

The current hybrid decoder profile showed substantial INT8 output-head work.
This disposable probe decreases the minimum Rayon chunk length by four for
dynamic INT8 matrix-vector output rows. It preserves each row's arithmetic,
the thread count, quantization, and the INT4 body. Production source is restored
before timing; isolated executables hold the two variants.

The ABBA test performs 512 identical teacher-forced decode steps per repetition,
with a 1,517-token prefix and the accepted hybrid overlay. Each process performs
one warmup and three measured repetitions, giving six measurements per variant.
Eight Rust workers and eight-thread BLAS environment limits match the prior
fixed-work comparisons. Builds and profiling do not overlap these measurements.
AC power is checked before each phase. The host is not otherwise isolated.

| Metric | Baseline | Finer chunks |
|---|---:|---:|
| Median decode including checksum | 3.404995s | 3.381622s |
| Observed decode range | 3.392587–3.458440s | 3.375499–3.412542s |
| Median prefill | 1.621127s | 1.624690s |

The 0.69% median decode reduction has overlapping sample ranges. This is not
strong enough evidence of a useful gain, so the scheduling change is not kept.
All sixteen repetitions produce the established all-logit SHA-256
`c129d845d9d0f6b2b1f7bc1aaa7741d1a9fb95b428a1016f519df94acb862f3a`.
All four processes exit normally without forced termination. No full-page or
llama.cpp speedup is claimed for this probe.

`abba/` retains the source patch, build records, binary identities, power
snapshots, raw timing/checksum outputs and process resources. `run.py` reproduces
the experiment from source commit `9387d0f` with fresh output and binary paths;
it deliberately refuses existing output paths. The accepted production CLI
remains the separately preserved NEON GELU binary.
