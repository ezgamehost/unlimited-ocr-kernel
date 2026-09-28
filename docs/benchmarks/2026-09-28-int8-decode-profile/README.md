# Current dynamic INT8 decoder profile

A three-second macOS `sample` capture from a resident dense-page request using
the same runtime binary and eight-worker/BLAS-cap configuration as the repeated
benchmark. The preceding dense request warms the model; capture starts 9.91314
seconds after the next request, based on the preceding first-token timing plus
0.15 seconds. Stack attribution, rather than that delay alone, establishes that
the capture includes decoder work. Warmup and profiled output token IDs match.
Both the profiler and inference process exited 0.

The collapsed top-of-stack list contains 4,069 samples in the dynamic INT8
four-row output closure, versus 102 in activation quantization. BLAS attention
work and Rayon scheduling also appear. These are aggregate thread samples,
including idle threads: they are not percentages of request wall time. The
large counts in wait functions must not be read as removable CPU work.

This supports testing the INT8 matrix-vector loop/layout next, with the existing
fixed-work decoder checksum and full-page quality gates. It does not yet prove
which change will improve latency. The older BF16 decoder profile is not used
as evidence for the current quantized bottleneck.

`process.json` retains the exact command, binary hash, sampling delay and exit
status. Child environment: `VECLIB_MAXIMUM_THREADS=8`, `OMP_NUM_THREADS=8`.
Request: dense corpus page, `<image>\ndocument parsing.`, cap 8192. The full sample
and both outputs are retained. Profiler overhead makes these timings unsuitable
for the benchmark speedup table.
