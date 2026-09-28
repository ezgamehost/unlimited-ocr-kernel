# Block-wise INT8 expansion

Weight expansion now traverses rows and 32-column blocks, loading each scale
once per block. This removes per-element quotient/remainder calculations while
preserving the exact signed INT8-to-FP32 multiply. Row tails and signed zero are
covered by bit-level checks against correctly rounded FP64 products.

The standalone benchmark expands each matrix 100 times, including allocation.
For 896×1280 and 1280×896 matrices, the previous implementation took about
0.063 seconds and the new one 0.009 seconds. The 257×513 tail case improved
from about 0.0073 to 0.0016 seconds. These operator gains are not whole-model
speedups; the retained raw logs include all samples.

Fresh-process dynamic-INT8 pilot, eight workers, Gundam mode, 2048-token cap:

| Page | Previous prefill (s) | New prefill (s) | New inference (s) | New process (s) | CER |
|---|---:|---:|---:|---:|---:|
| English | 3.314 | 2.011 | 12.121 | 18.605 | 0 |
| Chinese | 3.440 | 2.011 | 12.419 | 18.158 | 0 |
| Table | 3.349 | 2.115 | 12.863 | 18.635 | 0.003984 |
| Dense | 3.458 | 2.159 | 26.579 | 32.350 | 0 |

All four finish at EOS with exactly the preceding native token IDs. Startup is
included in process time. These single samples are not the final repeated
comparison: English and Chinese are only marginally below the earlier llama.cpp
fresh-process samples (18.738 and 18.474 seconds), while table and dense remain
slower. Dense emits 1649 native tokens versus 768 llama decode steps, so the
wall-time comparison also includes different output layout granularity.

An independent baseline parser bug was found while inspecting the raw llama
logs: only the first vision batch had been counted. The parser now sums every
batch and has a regression test reproducing the English log. Corrected totals
and an explicit correction record are in `../2026-09-28-llama-tiles32/`.
That correction changes vision-phase interpretation, not any process wall
time, output, CER or decode-time comparison.
