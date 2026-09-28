# Four-row block-packed INT8 experiment — rejected

Four output rows share each activation block; this candidate additionally stores
those rows' 32-byte weight blocks and scales consecutively. A scalar-oracle test
covers model widths, negative integer extremes, varied scales and unaligned
storage. Matrix tests cover packing twice, unchanged storage size, exact restored
weights, prefill equality, dynamic-decode equality and non-aligned fallbacks.
The packed representation replaces row-major storage in place using one four-row
scratch group. It does not duplicate the model. Only dynamic INT8 model loading
selects packing; weight-only/default BF16 behavior remains unchanged in the patch.

The operator pilot improved expert projections by about 25% and vocabulary
projection by about 20%. However, whole-decoder measurement exposed a prefill
regression that outweighed the small decode gain for the corpus output lengths.
The candidate is therefore reverted rather than enabled in production.

## Whole decoder evidence

The existing fixed-work benchmark runs the same 1,517-token reference prefix and
512 teacher-forced decoder steps. These are matched native-vs-native runs, not the
1,518-token llama OCR comparison. Baseline/candidate/candidate/baseline processes
run sequentially, each with one warmup and three measured repetitions, eight
workers, and VECLIB_MAXIMUM_THREADS=8 / OMP_NUM_THREADS=8. Vision and loading are
excluded. Decoder timings include the full-logit checksum.

| Median | Baseline | Packed candidate |
|---|---:|---:|
| 512 decoder steps | 3.738126 s | 3.621452 s |
| Prefill | 1.452696 s | 1.978876 s |

All sixteen repetitions have checksum 188943474296580907. Decode improves about
3.1%, but prefill costs another 0.526 seconds. A follow-up iterator-based
conversion loop still gives median prefill 1.986705 seconds in its three measured
pilot repetitions. It does not resolve the regression. That pilot is not a
second repeated A/B comparison and does not establish an additional speedup.
No full-page speedup is claimed from either variant.

`initial.patch` and `chunks.patch` retain both implementations and tests. RED
logs show the missing-kernel and missing-storage failures; release correctness,
full initial release suite and clippy logs retain passing checks. The follow-up
conversion passed the release quantized suite. `processes.json` records exact
benchmark commands, binary hashes and exits; all four exited 0. Initial timing
aggregates are in `initial-summary.json`; complete raw logs are retained.

The next experiment should avoid degrading prefill or should defer repacking
until its cost is justified. This result also shows why an operator speedup is
insufficient evidence for accepting a runtime optimization.
