# Rejected SAM crop batching experiment

Starting at commit 14b4708, attempted batches of three 640px local crops through
SAM, followed by per-crop CLIP/projector. All five vision tests passed against
the original reference, including ordering, partial batches and full merged
12-crop embeddings. No numerical tolerance changed.

Same-process ABBA timing for all 12 English-page crops:

| Path | Seconds |
|---|---:|
| Sequential | 9.225803 |
| Batch 3 | 9.255912 |
| Batch 3 | 9.238703 |
| Sequential | 9.198028 |

Batching adds memory use and complexity without a measured gain here. It is
not enabled or retained in production code. The exploratory patch and raw
measurements are saved so the result can be reproduced or revisited with a
different matrix backend. Apply `experiment.patch` on 14b4708, then run
`cargo run --release --example bench_crops` (requires pinned model/corpus).
