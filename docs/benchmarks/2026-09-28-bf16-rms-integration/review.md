# Scoped RMS implementation review

Reviewer bf16_rms_review found no blocking issue in the uncommitted scoped
implementation against 4aa5cc7. Input RMS, post-attention RMS, and final-logit RMS
all dispatch through the same decoder helper in prefill and decode. Every overlay
mode retains sequential arithmetic, including BF16 fallback projections. LayerNorm
and the existing rms_norm API retain their behavior. Pairwise summation uses O(n)
work, O(log n) stack, and no allocation.

The reviewer noted that successful fixed-work execution only prints a digest and
does not assert equivalence. The recorded four digests were subsequently checked
explicitly against the accepted value; q8-digest-equivalence.json records that
check. OCR output checks remain separate. The operator test covers accumulation
accuracy, not decoder dispatch by itself.
