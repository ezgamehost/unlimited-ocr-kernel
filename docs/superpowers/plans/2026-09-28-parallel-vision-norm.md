# Parallel vision normalization

Completed: full-vision ABBA 6.212014s→5.869081s (5.52% lower median latency),
all 16 embedding hashes identical; 59 Rust tests, original-model reference checks,
Clippy and formatting pass; all 12 OCR cases exact. Evidence and accepted CLI
identity are in `docs/benchmarks/2026-09-28-parallel-norm/`.

The accepted SAM-tile runtime's profile (`capture-sam1024`) contains 428 leaf
samples in normalization, which currently loops rows serially. All 3581 main
thread samples are under vision. These counts motivate an experiment; they are
not wall-time percentages or predicted speedups.

Keep scalar normalization APIs and arithmetic. Add a context-aware layer-norm
entry point that distributes independent rows only for sufficiently large
inputs and multiple non-scalar workers. Share row arithmetic with the existing
path, preserving FP64 centering/variance and FP32 outputs. Validation precedes
allocation/work; parallel execution adds no matrix-sized scratch buffers.
Route only vision's layer norms through the new entry point.

First verify bitwise parity with the current scalar operator on varied row
counts, widths, nearly constant values, bias/no-bias, and worker settings.
Run the full Rust suite and original model references. Build isolated full-vision
baseline/candidate executables, then time an ABBA sequence with identical
embedding hashes and no concurrent compute. Reject a measured regression.
If faster, require the complete 12-case OCR output gate, final formatting/Clippy,
and a verified preserved release executable before accepting the change.

The scalar fallback remains unchanged. No FP32 reduction shortcut, approximate
normalization, new device dependency, or decoder scheduling change is included.
