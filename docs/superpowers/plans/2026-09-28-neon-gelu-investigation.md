# Next investigation: preserve libm GELU arithmetic in a NEON fast path

Do not build, profile or run inference alongside `powered-abba-2` (exec80244).
Complete that comparison and its raw 96-request audit first. Preserve the
accepted parallel-normalization executable and vision benchmark baseline.

The post-SAM profile includes 1109 leaf samples in GELU, 427 in libm's erfc2 and
228 in libm's expf. These aggregate thread samples identify work to investigate;
they do not predict wall-time savings after the subsequent normalization change.
Check the updated profile before choosing whether to implement this candidate.

Local source inspected: pinned libm 0.2.16 `src/math/erff.rs`. In the ordinary
small-input range, erf uses FP32 rational polynomials and division. A NEON
implementation could evaluate four lanes with exactly the same coefficients,
separate multiply/add order and full-precision division, then perform the same
GELU outer arithmetic. Tiny inputs, exceptional values, larger magnitudes and
tails can retain the existing scalar libm path. Avoid replacing erf with tanh
or changing the mathematical approximation. Preserve the upstream permission
notice if adapting its implementation or coefficients into a derived routine.

Experiment sequence:

1. Profile the current accepted binary separately from benchmarks. If GELU
   remains material, add direct bitwise tests before the new SIMD implementation:
   signed zero, subnormals, interval boundaries and their adjacent floats,
   positive/negative values, mixed lanes, unaligned storage, all tails, and a
   broad deterministic set of finite bit patterns. Exceptional inputs retain
   scalar behavior. Test serial and existing Rayon scheduling without relaxing
   tolerances or changing decoder activations.
2. Start with the ordinary small-input interval; preserve the dependency's
   evaluation order and use scalar fallback for other lanes. Mask excluded
   lanes before speculative polynomial evaluation. No unchecked tail loads or
   approximate reciprocal instructions. Keep portable/scalar fallback intact.
3. Measure realistic activation buffers and full vision with isolated binaries,
   then an ABBA full-vision comparison with embedding hashes. Mixed-lane fallback
   overhead may erase gains; reject that outcome instead of assuming SIMD wins.
   Expand to another interval only as a separate measured hypothesis.
4. Acceptance requires the full Rust suite, original decoder/vision references,
   exact complete OCR on the 12 established cases, formatting/Clippy and a verified
   preserved final CLI identity. Report any bit/numerical/token change. Full-page
   system comparisons remain separate from operator/vision timing.

Implemented and verified: full-vision ABBA median 5.878761250 → 5.715417271 s,
all sixteen embeddings identical; twelve complete OCR cases exact. See
`docs/benchmarks/2026-09-28-neon-gelu/README.md`. Real model activations served
as the representative activation workload. Full-page rebenchmark remains pending.
Independent attention-head scheduling and
other measured bottlenecks remain alternatives if this experiment does not pay
off. Broader quality, precision/startup comparisons and final review remain in
scope; this plan does not replace them.
