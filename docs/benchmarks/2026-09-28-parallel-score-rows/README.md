# Parallel attention score rows

Profiling identified serial score normalization and SAM relative-position bias
as substantial vision work. Large score blocks now distribute independent rows
across the existing CPU pool, with at most roughly one partition per worker.
Blocks below 65536 floats and scalar/one-worker contexts retain serial execution.
Softmax preserves each row's reduction order and causal mask offset. Bias uses
reversed contiguous relative-position slices instead of repeated inner-loop
index calculations, retaining `score += vertical + horizontal` arithmetic.

Numerical tests require identical output bits for serial and parallel paths,
including causal offsets, uneven row counts and rectangular relative-position
grids. Invalid score layouts, causal bounds and NaNs are rejected. Both original
decoder reference tests and all three original vision reference tests pass at
their existing tolerances. The full Rust suite and Clippy were also rerun.

Eight-worker vision-only samples on the 12-crop English image:

- Parallel softmax alone: 8.010, 8.238, 7.866 seconds.
- Parallel softmax plus bias: 7.498, 7.526, 7.313 seconds.

These are sequential exploratory samples, not randomized final measurements.
The full-page pilot below starts a separate process per page, uses eight workers,
dynamic activation INT8, original Gundam preprocessing and a 2048-token cap.
All pages finish at EOS and have exactly the previous native dynamic-INT8 IDs.

| Page | Load (s) | Vision (s) | Inference (s) | Process wall (s) | CER |
|---|---:|---:|---:|---:|---:|
| English | 6.906 | 7.805 | 13.721 | 21.132 | 0 |
| Chinese | 5.842 | 7.608 | 13.929 | 19.888 | 0 |
| Table | 5.828 | 7.928 | 13.868 | 19.810 | 0.003984 |
| Dense | 5.846 | 7.620 | 27.642 | 33.618 | 0 |

The corrected llama.cpp fresh-process pilot was 18.738, 18.474, 17.821 and
20.113 seconds respectively. Native has not won that comparison. Both pilots
are single samples with differing quantization and attention policies; dense
generation also differs (1649 native tokens versus 768 llama decode steps).
This report includes startup rather than comparing resident native times to
fresh-process llama times. Raw output, commands, phase timings and process
statistics are retained in `parallel-score-fresh/`.
