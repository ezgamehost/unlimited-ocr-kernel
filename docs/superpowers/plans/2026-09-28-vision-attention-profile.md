# Next performance investigation: vision attention

Do not execute alongside the powered ABBA comparison (exec86420, output
`2026-09-28-packed-int4-resident/powered-abba-1`). The earlier exec33128 was
terminated for critical battery and is not live. Preserve the accepted binary
and complete all four powered phases before compiling, profiling or benchmarking.

Evidence: `docs/benchmarks/2026-09-28-packed-int4-resident/native-phase0-breakdown.json`.
In the completed native phase, vision takes about 7.1 s: 67–68% of English,
Chinese and table latency, and 52% of dense-page latency. This makes it a higher
priority than further small decoder gains. These are phase-0 observations, not
final ABBA results.

Static observations, not measured causes: `attention_bias` processes heads
serially and caps query blocks at128; each block calls two SGEMMs, scales/biases
scores and dispatches row softmax. SAM windowed attention has many small
196-token heads. Global SAM uses4096 tokens. Values transpose per head.

Prior evidence: `docs/benchmarks/2026-09-28-attention-blocks/README.md` already
tested raising the cap from32 to128, with about9% lower median global-view vision
time in three sequential samples. That change is already in the baseline; do
not repeat it or count its gain again. Crop batching was also tried and removed
for lack of a measured gain. Larger caps and head scheduling remain unmeasured.

1. Preserve current accepted executable identity and raw ABBA outputs. Profile
   repeated full-image vision on the existing original-weight reference input,
   separately record SAM/CLIP or operator contributions where needed. Do not
   attribute the 7.1s total to attention without measurements.
2. If dispatch/tile overhead is material, compare128/256/512/1024 query caps via
   isolated candidate builds, retaining scratch-limit-derived cap and fallback.
   Keep all arithmetic, positional bias and softmax semantics. Measure full
   vision/full-page effects, not just a favorable GEMM shape. Reject regressions.
3. If independent heads remain the bottleneck, evaluate bounded parallel heads
   with per-worker single-thread BLAS controls and aggregate scratch accounting.
   SAM bias callback currently uses FnMut; preserve existing public semantics
   and avoid sharing mutable closures unsafely. Older macOS/portable fallback
   remains valid. Do not stack Rayon over unrestricted threaded BLAS.
4. Each accepted change requires existing scalar/accelerated shape/tail/scratch
   checks, original global/local embedding fixtures, decoder/reference checks,
   complete OCR quality and repeated fair timing. Report numerical/token changes.

This investigation does not replace remaining precision/fresh-process validation,
broader quality checks or the required final whole-branch review.
