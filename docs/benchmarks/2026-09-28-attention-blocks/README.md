# Vision attention query-block experiment

Increasing the maximum query block from 32 to 128 reduces matrix-call count
and reuses matrix data across more queries. The implementation now also shrinks
the block to fit the configured scratch-memory budget rather than rejecting a
request that would fit with smaller blocks. Output and transposed-V storage
are included in the budget calculation.

Same-process global-view encoding samples, using the English corpus image:

| Maximum queries/block | Run 1 (s) | Run 2 (s) | Run 3 (s) |
|---|---:|---:|---:|
| 32 | 3.041284 | 2.979925 | 2.983292 |
| 128 | 2.717666 | 2.729064 | 2.702682 |

The median stage time improved about 9%. These are sequential exploratory
measurements of the full vision stage only, not complete OCR or a final
randomized comparison. `cargo run --release --example bench_vision` reproduces
the current path; set the query-block maximum back to 32 for the old variant.
No crop batching is enabled.

Validation: causal masking across block boundaries, an 8 KiB scratch budget,
original logo and English vision features, full 12-crop merged embeddings,
and the 300-position decoder reference all pass. No numerical tolerance was
relaxed. The separate crop-batching experiment had no measured gain and was
removed from production code.

A complete English-page pilot with eight workers reached EOS with exactly the
same 250 token IDs as the earlier native run and zero CER. Process time was
24.489 seconds: vision 11.248, prefill 8.298 and decode 3.675 seconds. The raw
record is `english.json`. This includes all recent scheduling improvements;
it is not an isolated end-to-end attribution to attention block size.
