# Current hybrid decoder profile

The accepted NEON GELU CLI was sampled during a warmed dense-page decode,
using eight Rust workers and eight-thread BLAS environment limits. The
command is the same as the audited full-page comparison; the binary identity
is verified against the new twelve-case exact OCR proof.

Both requests retain the audited dense output exactly and reach EOS. Inference
and profiler exit normally. The three-second capture starts after the preceding
request's first-token time plus 150 ms. Inspect the stacks for actual stage
attribution; the delay alone is not proof. Raw stacks contain decoder work.

Collapsed leaf counts include 1,821 samples in the INT4 matrix-vector output
closure and 1,510 in its INT8 counterpart, with 115 in activation quantization.
Attention BLAS work and scheduling are also present. These are aggregate thread
samples, including many wait samples, not wall-time percentages or a speedup
prediction. Profiled request timings include profiler overhead.

The next bounded probe gives Q8 matrix-vector scheduling four times as many
potential chunks, preserving each output's arithmetic. This tests load balance
against additional scheduling overhead without repeating the previously rejected
packed-head layout experiment. The Q4 expert path is unchanged. Acceptance
requires a measurable fixed-work improvement and exact output checks.

Reproduce with `profile.py` in the sibling `2026-09-28-vision-attention-profile`
directory, `--stage decode`, `--comparison` pointing to `powered-abba-2`,
`--quality` pointing to `2026-09-28-neon-gelu/quality`, and a fresh `--output`.
