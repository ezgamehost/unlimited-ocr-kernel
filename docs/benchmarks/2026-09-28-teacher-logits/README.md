# Full-decoder teacher-forced logits and RMSNorm precision

**Decision: rejected.** The higher-precision RMSNorm candidate improved the
logit comparison but failed raw-output preservation and slightly worsened
synthetic box accuracy. Production RMSNorm and the verified CLI
`f2433f3b6d50536de9990b3acdaa02e59ae749a9865ecf58d22a8d949ff4bfb8`
are restored. The new full-decoder test remains an explicitly open opt-in gate;
it still exposes the original prefill discrepancy. No tolerance was relaxed.

The final source review identified a missing acceptance check: existing tests
compared selected decoder layers, but did not consume the exported final logits.
The new opt-in `actual_decoder_teacher_forced_logits_match_upstream` test runs
the actual BF16 checkpoint through native prefill and 300 teacher-forced decode
steps, comparing all 129280 logits at each of the nine exported positions:
276, 277, 278, 404, 405, 406, 532, 533 and 576. This covers the 277-token prefix
and both 128-token ring wraps. Input embeddings and teacher-forced token IDs
come from the original CPU FP32 export; this test does not run image encoding.

The unchanged `1e-4 + 1e-4*abs(reference)` per-layer tolerance was initially
applied to the full logits. The accepted runtime failed on 24 values at the
prefill boundary: maximum absolute difference 0.0001540184, RMS 0.0000400269.
All eight decode rows passed and all nine highest-scoring token IDs matched.
`initial-logits.log` preserves this failure, not a successful parity claim.

`prefill_logit_component_diagnostic` then compared each cumulative layer's last
hidden row, each layer with independent upstream inputs, and the head using
the upstream final hidden row. Every hidden-row comparison passed the existing
tolerance. The head with upstream hidden state had maximum error 0.0000267029
and zero violations. The discrepancy therefore accumulates through prefill;
it is not isolated to the output projection. See `components.log`.

The separate `normalization.py` precision experiment applies upstream RMSNorm,
sequential FP32 reduction and an FP64 variance reduction to the exported
prefill inputs. The higher-precision reduction generally reduces discrepancy
against upstream. An independent numerical regression demonstrates the loss:
for one value of 10000 followed by 8191 ones, the old RMSNorm returns 90.509674
for the first component, versus the FP64 oracle's 90.50596139. Small squares
are lost when added to the accumulated large square in FP32.

The candidate retains FP64 precision only while reducing squared values,
rounds variance to FP32, then keeps the existing FP32 scale and affine
arithmetic. The standalone regression passes. The full-decoder test now passes
at the unchanged tolerance at all nine positions: prefill maximum difference
0.0001314878, RMS 0.0000235858, zero violations; all top token IDs still match.
The combined absolute/relative bound explains why the maximum absolute error
can exceed 0.0001 while every value passes. Raw results are in
`candidate-logits.log` and the red/green operator logs.

This was a candidate correctness change, not a performance
gain. Its full suite passed (67 standard tests, 12 opt-in tests ignored), as
do Clippy, four explicit decoder checks and eight vision checks. Candidate
CLI `4fda22acab9fa1104d5d3c65e0edc60a68949b035f533302277e4948a4d86c7f`
is preserved under `artifacts/kernel-baselines`; its source identity and raw
Q8 results are under `quality/`.

The Q8 exact-output gate **failed** after all twelve requests exited normally:
five cases have changed coordinates and token IDs. All recognized text, CER,
token counts and EOS match, but raw-output equivalence does not. The independent
`coordinate_diff.py` check masks only coordinate tuples, preserving all other
markup and characters; it confirms that all differences are confined to boxes:

| Case | Changed boxes | Largest coordinate difference |
| --- | ---: | ---: |
| Chinese, matched prompt | 1 | 1 |
| Dense, original prompt | 27 | 1 |
| Receipt, original prompt | 1 | 3 |
| Bilingual notice, original prompt | 3 | 1 |
| Archive register, original prompt | 20 | 2 |

These are emitted model-coordinate units. `coordinate-differences.json` retains
every changed tuple. The failed gate stopped before hybrid execution; the
separate hybrid gate under `quality-hybrid/` also failed exact preservation in
five of twelve cases, with all recognized text and zero CER preserved. Both
candidate inference processes exited normally; the Python gates exited 1 on
their exact-output assertions. Neither CER equality nor improved logit error
is being substituted for an exact-output pass.

The independent `layout_score.py` uses the frozen held-out manifest's rendered
line bounds and reconstructs primary-corpus bounds from the original recipe,
checking font hashes and Pillow version. It aligns every output block with
contiguous recognized source lines and scores their union for grouped regions.
Coordinate conversion follows upstream drawing (`int(value / 999 * dimension)`).
Raw per-block results and manifest hashes are in `layout-scores.json`.

| Q8 case | Mean box IoU, before → candidate | Mean edge error in pixels, before → candidate |
| --- | ---: | ---: |
| Chinese matched | 0.810406 → 0.805694 | 4.6250 → 4.6667 |
| Dense original | 0.758709 → 0.755354 | 2.4649 → 2.4868 |
| Receipt | 0.732510 → 0.732158 | 3.0000 → 3.0313 |
| Bilingual notice | 0.751419 → 0.747159 | 4.1563 → 4.2031 |
| Archive register | 0.780628 → 0.775108 | 3.5787 → 3.6389 |

The changes are small, but both metrics worsen in every changed Q8 case. This
is synthetic typographic layout evidence, not a production-document benchmark.
It supplies an independent reason to reject the change instead of judging
raw coordinate differences solely by equality with the previous binary.
`candidate.diff` preserves the rejected RMSNorm implementation and its numerical
regression test; the active runtime and operator tests have been restored.

Exported expert IDs/weights still require an explicit audit;
this logit check does not silently substitute for that requirement. No claim
is made that the change repairs the separately documented hybrid max32 failure.

Reproduce the opt-in full-logit check after preparing original reference files:

```sh
cargo test --release --test decoder actual_decoder_teacher_forced_logits_match_upstream \
  -- --ignored --exact --nocapture --test-threads=1
```

The reference exporter and artifact paths are the existing ones in
`tools/reference.py`. The numerical Python experiment is offline validation;
the production inference runtime remains native Rust.
