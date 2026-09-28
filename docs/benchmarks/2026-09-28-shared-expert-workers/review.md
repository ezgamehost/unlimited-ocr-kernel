# Independent read-only review

Reviewer: `/root/shared_expert_review`, production diff against `1d9b8db` in `src/cpu/mod.rs` and `src/decoder/mod.rs`. No Critical, Important, or Minor findings.

The review confirmed reuse of the existing Rayon pool, independent output-row computations, ordered expert accumulation, unchanged scratch partition and low-memory fallback, and safe saturation for smaller worker pools. Scalar behavior remains serial, while BF16/Q8/Q4 retain their per-row arithmetic. The AC fixed-work records and source identities support the measured 2.22% Q8 decoder improvement. The baseline-to-HEAD commits contained only benchmark documentation.

The reviewer performed no builds or tests. Runtime acceptance depends on the separately recorded validation and complete OCR gate. The known baseline logit-parity and Q4 stress-case accuracy issues remain open.
