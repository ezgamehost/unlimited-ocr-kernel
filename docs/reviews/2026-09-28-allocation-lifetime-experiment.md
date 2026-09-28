# Allocation-lifetime experiment plan and outcome

Baseline source: `a36fc58`. Candidate changes only `Model::infer`: release prepared
pixels after vision encoding, projected features after copying them into prompt
embeddings, and prompt embeddings after decoder prefill. None is used again.
All releases remain inside their corresponding stage timer. This section records the proposed experiment; the completed outcome below
rejects the candidate as a default performance improvement.

For the four-image corpus, prepared pixels occupy 71,565,312 bytes (68.25 MiB),
projected features 7,746,560 bytes, and matched prompt embeddings 7,772,160 bytes.
For the 32-crop portrait those payloads are 169,869,312, 19,420,160, and
19,445,760 bytes respectively. These are disjoint owned FP32 tensors. Releasing
them earlier reduces overlapping live payloads, but allocator reuse and earlier
vision peaks may prevent an equivalent peak-RSS reduction. No RSS gain is claimed.

Validation must run after the active BF16 llama resident benchmark so compilation
and inference do not contend with its timed requests:

1. Preserve the candidate diff; build the baseline and candidate from otherwise
   identical source, recording source and executable hashes.
2. Run the full Rust suite and clippy. Check original decoder/vision references.
3. Compare complete four-image outputs (token IDs, text, EOS, CER) under the same
   prompt, dynamic INT8 overlay, eight workers and eight BLAS threads. Include
   baseline/candidate order reversal and retain all raw measurements.
4. Re-run the 32-crop resource input with the same cap and compare exact child
   peak RSS and outputs. Its known token-limit OCR failure remains a failure;
   resource savings cannot turn it into a successful OCR result.
5. Retain the change only with correct output and evidence supporting its cost
   and memory behavior. Finish broader precision/fresh-process/held-out work.

## Held-out runner prerequisite

The current resident runner validates exactly 1,513 image tokens / 1,518 prompt
tokens for every image. This correctly enforces the existing four-page corpus,
but cannot admit held-out aspect ratios that produce a different crop grid.
Before adding such measurements, make the expected image-token count explicit
per manifest item, preserve the current count for existing manifests, and check
both native image tokens and server prompt tokens against the declared count.
The fixed matched prompt contributes five non-image tokens. Record per-image
expectations in process metadata. Regression coverage should exercise a real
driver response for a 273-token single view and reject mismatches; do not merely
remove the current checks. Do not change the runner while the active BF16
process is using its recorded hash. This is a pending prerequisite, not a finding
that invalidates the current fixed-shape measurements.

## Outcome

Completed and reverted. Full tests, original references, all 32 corpus outputs
and all four 8,192-token extreme-input outputs passed their checks. Earlier
release reduced maximum-crop peak RSS in both orderings, but increased observed
four-page process peak RSS and did not establish a consistent latency benefit.
See [the retained experiment](../benchmarks/2026-09-28-allocation-lifetime/README.md).
The source is restored; allocation reuse and broader resource work remain in
scope. This experiment does not resolve the known 32-crop OCR failure.
