# Parallel vision layer normalization

Vision layer normalization now distributes independent rows across the existing
CPU worker pool for large inputs. Each row retains the original FP64 mean and
variance calculation, operation order, affine transform and FP32 output. Scalar
and single-worker execution remain sequential. Validation still precedes output
allocation, and parallel execution adds no matrix-sized temporary buffers.
The existing scalar layer-norm and RMS-norm APIs keep their behavior; only
vision calls the new context-aware layer-norm entry point.

The accepted SAM-tile runtime's updated profile exposed serial normalization
work: 428 normalization leaf samples in the
[SAM-tile profile](../2026-09-28-vision-attention-profile/capture-sam1024/top-stacks.txt).
These aggregate samples motivated the experiment, not a
predicted speedup. No profile ran during the timing comparison.

## Measured result

| Full English Gundam vision stage | Median (s) | Observed measured range (s) |
|---|---:|---:|
| Accepted SAM-tile baseline | 6.212014 | 6.187437–6.230300 |
| Parallel normalization | 5.869081 | 5.855813–5.890419 |

The median vision latency reduction is **5.52%**. `abba/` contains four
sequential processes in baseline/candidate/candidate/baseline order. Each
process performs one warmup and three measured encodes, giving six measured
values per variant. All 16 outputs have shape [1513, 1280] and identical FP32
embedding SHA256:
`a2b4ecf2092508904b5bd359c86fbce682c801ffd7cd93336658b25c7acc7213`.
The finite check and hashing occur outside the encode timer.

Configuration: shared M4 Pro, AC checked before each process, eight Rust workers
and eight BLAS threads, original vision weights. Executable/model/source hashes
and the candidate source patch are retained in `abba/`. No compilation,
profiling or other inference ran alongside these measurements. This is a
vision-stage result; a new full-page llama.cpp comparison remains pending.
Do not combine percentages from separate experiments into a measured total.

## Validation

- New bitwise-parity test: varied rows/widths, nearly constant values, bias and
  no bias, scalar backend, and one/four accelerated workers. Its first run
  failed to compile because the new API was absent; after implementation the
  full Rust suite passed 59 tests, with 9 intentionally ignored.
- The original decoder and vision checks were run explicitly: 2 decoder and
  8 vision checks passed, including global and merged crop embeddings. No
  numerical tolerance changed.
- Clippy with warnings denied and formatting checks passed.
- All 12 complete OCR cases matched accepted hybrid text, token IDs, EOS and
  image-token counts exactly. This preserves their existing CER 0 results on
  eight synthetic selection pages and four matched-prompt primary cases; it
  does not establish accuracy on unseen production documents.
- The final release executable was checked against the OCR run and copied to
  `artifacts/kernel-baselines/uocr-<SHA>`. Accepted SHA256:
  `7baaad44d8ae6f56fd6f470d29271a0b0529be83a8eacd53b69d60c9140c08ba`.

`run.py` reproduces the ABBA protocol using the preserved SAM-tile baseline and
the current `bench_vision_tiles` example executable. It requires fresh artifact
and output paths, validates binaries/model, and rejects embedding drift. Run
from the repository root with `PYTHONPATH=. .venv/bin/python` after building
`cargo build --release --example bench_vision_tiles`. Existing measurements
are deliberately protected from overwriting.

`quality/` retains the complete OCR evidence, `accepted.json` the final identity,
and the test/build logs the validation commands' output. The final CLI build
occurred after tests, preventing Cargo's integration-test build from replacing
the executable that received the OCR validation.
