# Vision profiling protocol

`capture-1` completed after the powered 96-request audit. The profiler and
inference process both exited 0, and both OCR outputs exactly matched baseline
text/token IDs and reached EOS. The accepted binary is unchanged.

Actual capture bounds were 0.210–5.494s after request submission. All 3578 main
thread samples are under `Vision::encode`, establishing vision attribution.
The collapsed leaf table includes 978 samples in GELU, 362 in libm's erfc2,
686 in softmax, 579 in vector exp, 367 in normalization, 250 in relative bias,
and 207 directly in SAM's attention loop, alongside BLAS and many waits.
These are aggregate thread samples, not wall-time shares or predicted gains.

Attention dispatch, softmax, and activation work all warrant investigation.
The next experiment compares larger attention query tiles using isolated
executables and full-vision measurements. GELU remains a separate candidate;
this capture does not establish that any particular change will be faster.

To reproduce after a completed powered 96-request audit, use a fresh output
directory (the example's `capture-1` now exists):

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-vision-attention-profile/profile.py \
  --comparison docs/benchmarks/2026-09-28-packed-int4-resident/powered-abba-1 \
  --output docs/benchmarks/2026-09-28-vision-attention-profile/capture-1
```

The script requires AC power and a fresh output directory, reuses the accepted
binary and eight-worker/eight-BLAS-thread configuration, warms an English page,
then samples the second request for five seconds starting shortly after submission.
Both outputs must match the baseline text and token IDs exactly and reach EOS.
It records the actual capture bounds, raw stacks, responses, process identity,
and child resource/exit accounting.

The planned delay targets vision based on its measured seven-second duration.
Confirm stage attribution from stacks; timing estimates alone do not prove it.
Thread sample counts include waits and are not percentages of request latency.
Profiling timings must not enter the benchmark speedup table.
