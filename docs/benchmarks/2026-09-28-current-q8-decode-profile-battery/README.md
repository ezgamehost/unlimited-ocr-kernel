# Current Q8 decode profile — battery diagnostic

The restored, verified CLI `f2433f3b6d50536de9990b3acdaa02e59ae749a9865ecf58d22a8d949ff4bfb8`
completed two dense-page requests with exact text and token IDs from the audited
Q8 comparison. Both reached EOS at 768 emitted tokens. The native process and
sampler exited normally; no forced kill occurred.

This is a three-second stack capture on battery in normal power mode, above
50% charge, with power snapshots retained before and after. It is exploratory
attribution, not a performance comparison. Profiler overhead and battery
conditions preclude treating these page timings as a new benchmark.

Of the 1835 main-thread samples, 1812 are in `Decoder::step`, with 22 in token
selection. Across sampled threads, the collapsed output-computation closure
for dynamic INT8 has 4903 leaf samples, versus 131 for activation quantization.
There are also BLAS attention and substantial worker-wait samples. These
aggregate thread samples are not wall-time percentages.

The retained disassembly of the sampled Q8 closure shows four row-major scale
values gathered into vector lanes for every 32-column block. Frequently sampled
offsets include 1428, 1476 and 1500; this does not prove a memory-bound kernel,
but motivates testing a more direct scale-load path. A bounded next candidate
is an interleaved scale cache for the vocabulary head only, preserving the
original weight layout, FP32 multiply/add order, and prefill storage. Its extra
20.7 MB and preparation cost must be counted. Earlier full-weight packing and
eight-row trials were rejected; this is a distinct hypothesis, not evidence
that the proposed cache will improve performance.

Reproduce with the existing profile runner, a fresh output directory,
`--stage decode --power-source battery`, the completed current-Q8 comparison
and `2026-09-28-final-review/q8` exact-output proof. AC remains the default;
battery mode requires explicit selection. All authoritative full-page timing
continues to use AC power.
