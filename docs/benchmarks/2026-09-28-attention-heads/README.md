# Parallel single-token attention heads

Base: `7716258`. The initial candidate patch parallelizes independent heads in
`attention_decode`, preserving the original strided matrix-vector calls,
softmax, and each head's arithmetic. The prefix/ring attention policy is
unchanged. Large enough inputs with multiple workers use parallel execution
only when the aggregate score buffers fit the scratch budget. Otherwise the
original single-score-buffer serial path remains available.

The operator regression compares output bits at one and eight workers for
129- and 1,641-token caches, with enough scratch for all heads and with exactly
the original minimum scratch. One byte below that minimum remains an error.
The FP64 oracle and malformed-layout checks remain in place.

The same resident fixed-work benchmark as preceding reports compares 512
teacher-forced decoder steps, a 1,517-token prefix, and eight workers. It is
not an OCR completion benchmark or a llama.cpp comparison. Each process loads
once and performs one warmup plus three measured repetitions. Processes run
base/candidate/candidate/base, followed by another candidate/base pair because
the first candidate process showed substantial timing variation. Every sample
is retained, including warmups; no outlier filtering is applied.

The host is not isolated. During the first candidate run a process snapshot
showed an unrelated container using approximately one CPU core, plus other
activity. The benchmark driver itself ran no concurrent builds or inference.
These measurements support an exploratory implementation decision, not a
final hardware speedup claim. `process-snapshot.txt` records nearby activity.

Across nine measured samples per version, median decode duration is
4.473741 seconds for the base and 3.885156 seconds for the candidate (13.2% less
elapsed time). All 24 repetitions retain logits checksum
188943474296580907. See `fixed-work.json` for all individual values.

## BLAS thread policy correction

The measurements above and `fresh/` precede the BLAS guard and are preliminary.
The accepted candidate must also prevent Accelerate from creating nested BLAS
workers inside the parallel head tasks. The SDK's `thread_api.h` and Apple's
[BLASSetThreading documentation](https://developer.apple.com/documentation/accelerate/blassetthreading%28_%3A%29)
specify a thread-local policy. A non-Send RAII guard selects single-threaded
BLAS around strided GEMV and restores the previous setting on every exit.
The macOS 15+ entry points are dynamically resolved once; missing symbols on
older systems disable parallel Accelerate heads, retaining serial execution.
Portable non-Accelerate head execution does not require this API.

The guard test was first observed failing to compile without the implementation,
then passed with the implementation. It verifies nested restoration and unwind
restoration. Its deliberately caught panic is expected in the raw log. The
release log confirms that the API is present on this machine. Three attention
checks and two original-weight decoder reference tests pass with the guard.

Dense/prefill SGEMM still uses Accelerate's own threading policy. An eight-worker
Rayon setting must not be described as a global eight-thread BLAS limit. The
broader threading-policy audit and matching final benchmark controls remain open.

Guarded candidate retest: fresh base/candidate/candidate/base processes, each
one warmup plus three measured repetitions. All sixteen checksums remain
identical. Six measured samples per version give medians 4.059052 seconds
(base) and 3.766247 seconds (guarded candidate), 7.2% less elapsed time.
`guarded-fixed-work.json` and the associated raw logs preserve every sample.

Final guarded source passes 43 ordinary Rust tests (eight artifact-dependent
tests ignored by default) and clippy with all targets and warnings denied.
The original-weight decoder tests and fixed-work benchmark were also run
explicitly as documented above. `candidate.patch` is the guarded version;
`unguarded-candidate.patch` preserves the earlier experiment.

## Guarded resident OCR pilot

One process loads once, then runs four independent requests sequentially,
with eight Rust workers, dynamic INT8 decoder and the full 8,192-token cap.
No prompt/KV reuse occurs between requests. All outputs reach EOS and retain
exactly the prior token IDs. CER is zero except the existing one-character
table error (0.003984). This is one pilot per image, not the final repeated
resident comparison. Process-wide resource information is in
`guarded-resident/process.stderr`; load is reported separately in `process.json`.

| Page | Request wall (s) | Vision (s) | Prefill (s) | Decode (s) |
|---|---:|---:|---:|---:|
| english | 11.630 | 7.617 | 2.026 | 1.842 |
| chinese | 10.836 | 7.147 | 1.432 | 2.117 |
| table | 10.519 | 7.135 | 1.438 | 1.809 |
| dense | 21.346 | 7.201 | 1.424 | 12.464 |

Reproduce from the repository root with
`PYTHONPATH="$PWD" .venv/bin/python docs/benchmarks/2026-09-28-attention-heads/run_resident_pilot.py`.
The script rewrites its pilot output directory.
