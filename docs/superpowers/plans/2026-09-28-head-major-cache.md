# Head-major decode cache candidate

The isolated stride probe showed a lower attention time with identical bits.
It excludes packing cost and does not prove a decoder or page speedup. Test the
complete runtime before accepting a layout change.

Preserve the existing token-major LayerCache constructor/API by default. Add
an explicit conversion after each runtime prefill layer to head-major storage,
with capacity prefix+window per head. Keep active token count separate from
capacity. A push updates the same physical ring slot in every head, preserving
the original score/reduction order and absolute position. No second permanent
KV copy. Conversion allocation failures must leave the original cache intact.

Share single-query attention arithmetic between validated token-major and
head-major wrappers. The latter receives the active count and uses the fixed
capacity to locate each head, ignoring unused slots. Keep scalar fallback,
thread-local BLAS policy and existing scratch limits. Runtime prefill still
uses the existing token-major path, then converts the completed cache.

1. Add cache-layout/ring and attention parity tests. Observe missing-API RED.
2. Implement conversion and shared attention access. Test partial ring fill,
   multiple wraps, malformed shapes, active count, minimum scratch and threads.
3. Run suite, then fixed-work baseline/candidate/candidate/baseline. Include
   conversion in prefill timing and verify the full logits SHA. Reject a
   regression or inconclusive gain; preserve the experiment.
4. For a useful candidate, run explicit original decoder/vision references,
   formatting/Clippy, final CLI build and twelve exact OCR cases. Preserve the
   quality-verified binary. Full-page comparison remains a separate gate.

Standing user authorization is to continue optimizations without questions;
this bounded candidate does not change prompts, model precision or cache policy.

Progress: all correctness gates pass, including full-logit and twelve OCR cases.
The AC series stopped before its first phase when power disconnected. A separately
labeled battery ABBA completed with 4.08% lower median decode and exact logits;
candidate retained for AC confirmation. See the benchmark README for raw evidence.
Ruling: run correctness and separate battery exploration while AC is unavailable;
these do not substitute for the AC/full-page validation requirements.

AC confirmation completed: all16logit hashes exact; median512decode4.48%lower,
prefill+20.1ms. Layout accepted with measured conversion cost. CurrentBF16/Q8
max32outputs also match older outputs exactly; hybrid max32fails identically
with both layouts. Full-page update remains pending.
