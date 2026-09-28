# NEON GELU experiment

The small-input erf polynomial now evaluates four lanes together on AArch64,
preserving libm 0.2.16 coefficients, separate multiply/add order, full division,
and the existing GELU outer arithmetic. Other ranges and incomplete vectors
retain scalar evaluation. Scalar backend behavior is unchanged.

The isolated full-vision ABBA experiment compares the accepted parallel-norm
baseline with this candidate. Each of four processes performed one warmup and
three measured encodes, giving six measurements per variant. Median vision
latency decreased from 5.878761250 to 5.715417271 seconds (2.78%). Every one of
the sixteen embeddings had the same SHA-256. This is a stage measurement on
one image, not an updated full-page comparison against llama.cpp.

See `abba/summary.json` for every observation, `abba/identity.json` for source
and binary identities, and `run.py` for the measurement procedure. Real model
activations were measured directly rather than synthetic activation buffers.

The recorded Rust suite passed 60 tests with nine ignored; the explicitly run
original-model decoder and vision references also passed, as did Clippy. The
new direct test covers interval boundaries, arbitrary float bit patterns,
mixed lanes, offsets and tails. Finite outputs are checked bitwise against
scalar libm; NaNs retain the existing NaN-class contract. Raw verification
logs are retained here. The separate `quality/` run checks twelve complete OCR
cases against frozen outputs; only a completed comparison with all exactness
flags true and a normal process exit establishes that gate.

That gate completed successfully: all twelve cases have exact text, token IDs,
EOS termination and image token counts. The process exited normally. The CLI
hash was rechecked against the quality run and the executable preserved; see
`cli-identity.json`. Formatting and `git diff --check` also passed. A new
full-page llama.cpp comparison incorporating this change remains pending.
