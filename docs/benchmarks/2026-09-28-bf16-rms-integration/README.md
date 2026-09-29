# Scoped unquantized-decoder RMS candidate

Status: accepted as a scoped numerical correction after correctness/output checks
and an isolated operator-cost check. Model-level timing remains inconclusive;
the published full-page benchmark headline is unchanged. The initial implementation
patch is retained as source.patch.

Decoder input, post-attention, and final-output normalization select balanced FP32
RMS reduction only when no quantized overlay is attached. Any overlay keeps the
previous sequential arithmetic, including fallback unquantized projections.
LayerNorm and the existing rms_norm public operator keep their prior behavior.
The new additive rms_norm_pairwise operator shares validation and uses no extra
heap allocation. Leaves contain at most sixteen elements; recursion is logarithmic.

Validation completed normally:

- All standard Rust tests, fmt, and clippy with warnings denied pass.
- The targeted accuracy test was first observed failing for the missing API;
  the prior global experiment also records its numerical failure on scalar RMS.
- All nine teacher-forced logit checkpoints pass the unchanged tolerance,
  including prefill and two ring-wrap boundaries.
- Existing independent decoder-layer and expert-scheduling references pass.
- Both eleven-layer routing reference gates pass.
- All four complete Q8 decode-logit SHA256 values explicitly match the accepted
  `3e4fa2e2d048750fba24ad7cde61ccc7c92e1bc2c7436e0d387d867387c6c7da`.
- All twelve Q8 and twelve hybrid OCR outputs exactly match their accepted
  baselines, including coordinates, token IDs, EOS, image count and precision.
- All twelve scoped BF16 OCR outputs also exactly match the accepted BF16 run,
  including coordinates, token IDs, EOS, image count and precision.

All three OCR child processes exit normally without forced kills. The first BF16
wrapper invocation had a syntax error before process creation; its log is retained.
After correcting the extra parenthesis, the full BF16 wrapper completed normally.
These are battery-powered correctness runs: timing fields are not speed evidence.

Source review found no blocker; review.md records the findings and the explicit
digest verification added in response. validation/steps.json and logs record
commands/exits. Candidate CLI SHA256:
`0268d10cf8fb35c668282d3cbf68648e583bbdc30dd0f7fd4e96a0c5cda014b0`.
The preserved candidate unit-test binary and hash are in candidate-unit.json.
CLI `05616a6d...` is the prior accepted baseline used in these comparisons.

Run validate.py via the project Python environment and PYTHONPATH=. to repeat
correctness gates, using fresh output directories. The Q8/hybrid checks use
../2026-09-28-final-review/quality.py --output-root pointing into this directory.
bf16_quality.py compares the scoped CLI against the completed, hash-identified
accepted BF16 baseline from ../2026-09-28-pairwise-rms/bf16-quality.

## Battery-only ABBA regression screen

run.py completed baseline/candidate/candidate/baseline with four repetitions per
process, the first designated warmup. Power remained battery, charge above 50%,
and power modes zero at every before/after check. Both binaries use the same
fixed-work fixture and test-only route-capture harness. All four children exited
normally, and each variant's full-logit digest was stable across its eight runs.
Different BF16 digests between variants are expected from the accuracy change;
correctness is established by the separate reference/output gates above.

| Measured samples (six each) | Baseline | Candidate |
|---|---:|---:|
| Decode median seconds | 4.955708 | 5.972002 |
| Decode min–max seconds | 4.610451–41.183333 | 4.814905–35.565091 |
| Prefill median seconds | 1.826261 | 2.452786 |
| Prefill min–max seconds | 1.423682–8.390955 | 1.516052–9.371330 |

These raw results do not establish performance acceptance. The final baseline
phase also develops severe stalls, and the measured ranges are extremely broad.
All outliers remain in the summaries. The post-run host snapshot shows about
10.9 GiB swap in use and no reported thermal warning, but without start/end memory
deltas this does not attribute the stalls to paging. This is an exploratory battery
screen, not comparable to the existing AC headline and not a speedup claim.

This screen did not justify accepting or rejecting the candidate on throughput.
The following resident operator measurement resolves the narrower implementation
cost question; model-level throughput still needs stable-host confirmation.

## Resident operator-cost check and acceptance

The ignored resident_rms_reduction_comparison test alternates both public RMS
operators in one process, using 1280-wide rows, ten warmup calls per phase, and
ABBA then BAAB order. It includes the same validation, output allocation and
normalization in both paths; black_box consumes every result. All 24 phases finish.

| Rows per call | Sequential median ns/row | Pairwise median ns/row |
|---|---:|---:|
| 1 | 2069.47 | 1800.27 |
| 277 | 1651.97 | 1464.40 |
| 1517 | 1732.82 | 1425.58 |

All three observed ranges are non-overlapping in this battery-only operator run.
This supports no intrinsic normalization-cost regression on the tested shapes;
it does not establish model speed or explain the earlier model stalls. The
11–18% lower operator times must not be extrapolated to whole-page throughput.

Acceptance is for a numerical correction: unchanged logit tolerance now passes,
36 OCR outputs remain exact, quantized full-logit digests remain exact, source
review is clear, and isolated operator cost improves. The evidence does not claim
a new llama.cpp win, a reliable model-level speedup, or closure of broader accuracy
and performance requirements. A stable model-level rerun remains outstanding.
