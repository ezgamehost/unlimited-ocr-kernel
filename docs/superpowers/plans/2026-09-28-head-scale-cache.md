# Vocabulary-head scale-cache experiment

Hypothesis: the current four-row INT8 kernel spends enough work gathering
row-major block scales that a contiguous four-scale load can improve decode.
The Q8 battery profile has 4903 collapsed leaf samples in output computation,
and the retained assembly shows the scale gather. This is motivation, not a
speedup prediction. Earlier weight-layout and eight-row experiments were
rejected; do not repeat them.

Scope: keep weight bytes and canonical row-major scales unchanged. Prepare
only the Q8 vocabulary head's scales in `[row_group, block, four_rows]` order
for dynamic-activation inference. This adds 20684800 bytes for the current
129280-by-1280 head, plus a one-time preparation cost. Preserve prefill and
weight-only access to canonical scales, portable fallbacks, storage accounting,
and all existing artifact validation. No serialized format change is needed.

Implement a checked, fallible preparation method and a specialized four-row
load path sharing the original arithmetic: integer block sums, conversion,
weight-scale multiply, activation-scale multiply and ordered FP32 addition.
Do not introduce FMA or change block reduction order. Unsupported shapes must
retain the existing path. Enable preparation at the model's dynamic-activation
setup so repeated requests reuse it and ordinary activation quantization does
not repeatedly prepare caches.

Before implementation, preserve a freshly built baseline unit executable for
the fixed-work Q8 decoder test. Add bitwise comparisons against the existing
kernel with varied blocks, unaligned slices, zero scales and row/tail cases.
Test prepared/unprepared complete matrix outputs and account for cached bytes.
Demonstrate the missing API/test failure before implementation.

Measure the actual fixed-work model in ABBA order with identical 1517-token
prefix and 512 teacher-forced steps, eight workers and eight BLAS threads.
Require all full-logit hashes to match between baseline and candidate. Record
prefill, preparation/startup and process memory alongside decode. Battery trials
may be separately labeled exploratory with explicit power guards; require AC
confirmation for accepted speed claims. Reject insignificant gains or material
prefill/memory regressions; never infer a full-page win from the kernel alone.

If the candidate is useful, run standard tests, original passing component
checks, formatting and Clippy, then build the final CLI and require exact Q8
and hybrid OCR output gates. The known full-decoder prefill tolerance failure
remains open and must be unchanged, not hidden by this optimization. Preserve
the final binary and source identity before any new full-page comparison.
