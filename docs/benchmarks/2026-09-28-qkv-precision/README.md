# QKV scheduling precision coverage

The existing concurrent QKV implementation preserves all fixed-work decoder
logits in BF16 and weight-only Q8. These two precision modes do **not** show a
clear performance improvement in this series. No production dispatch changes
were made from these results.

| Precision | Sequential median (s) | Concurrent median (s) | Median change |
| --- | ---: | ---: | ---: |
| BF16 weights / FP32 activations | 4.826730 | 4.905901 | 1.64% slower |
| Q8 weights / FP32 activations | 4.920079 | 4.847144 | 1.48% faster |

Each measurement covers 512 teacher-forced steps after a 1517-token prefill,
including hashing every output logit. Both modes use eight CPU workers and eight
BLAS/OMP threads, AC power, normal power mode, and baseline/candidate/candidate/
baseline process order. Each process records one warmup and three measurements,
for six measured samples per variant and precision. All eight processes exit
normally without forced kills. The entire 16-repetition logit digest agrees
within each precision; equality between different precisions is not claimed.

Ranges overlap: BF16 baseline 4.695–5.176 seconds versus candidate 4.631–4.973;
weight-only Q8 baseline 4.787–5.090 versus candidate 4.807–5.008. Baseline phases
drift in opposite directions across the two experiments. BF16's initial warmup
is 10.462 seconds; it is retained and excluded from measured medians under the
predeclared warmup rule. This shared-host experiment does not isolate startup,
paging, or thermal effects. It does not establish regression or improvement
for these modes. The separately measured dynamic-INT8 gain must not be
extrapolated to all precisions.

`build.py` builds the same extended benchmark harness twice, using sequential
QKV source from `21da757` and concurrent QKV source from `a3a53e1` (also present
at evidence commit `77c19d5`). It restores production source in a `finally`
block, records binary/source identities, and verifies that the accepted CLI
has not changed. `harness.diff` contains the test-only extension supporting
`UOCR_BENCH_PRECISION=bf16|weights|dynamic`; the default remains dynamic INT8.
BF16 loads original weights without a quantized overlay. The runner validates
precision labels, step counts, prefix length, process exits and all-logit hashes.

This closes fixed-work scheduling equivalence coverage for these two modes,
not full-model upstream parity or broad OCR accuracy. The known BF16 prefill
logit tolerance failure remains open. Whole-page performance remains documented
in [the separate 96-request comparison](../2026-09-28-q8-scheduling-fullpage/README.md).

Reproduce from the repository root with `build.py`, then:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-qkv-precision/run.py --precision weights --output-name weights-new
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-qkv-precision/run.py --precision bf16 --output-name bf16-new
```

Use fresh output names, AC power, the pinned artifacts, and no concurrent heavy
work. Raw logs, per-process resource records and summaries are retained.
