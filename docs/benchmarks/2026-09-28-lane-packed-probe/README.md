# Rejected lane-interleaved Q8 layout experiment

**The production runtime is restored unchanged.** Four-byte-lane-interleaved
weights improve the isolated INT8 dot kernel, but three full-decoder experiments
retain a prefill cost that makes the small decode gain insufficient to accept
without demonstrated whole-page benefit. This is a rejected candidate, not a new
headline speedup or a proof that every document would be slower.

The proposed layout interleaves four-byte chunks from four output rows. Eight
`SDOT` lane instructions accumulate a 32-element block directly into four output
lanes, avoiding the old horizontal reductions. Per-block FP32 multiplication and
addition order is preserved. Packing uses one four-row temporary and rewrites
owned weights/scales in place. Serialized artifacts and total weight storage are
unchanged; BF16 and weight-only loading retain their original layout.

## Correctness and isolated measurements

The initial test fails because the two proposed helpers do not exist (`red.log`).
The implementation then passes bitwise dot/dequantization tests for widths
32/64/96/896/1280/6848, input offsets 0–3, signed INT8 extremes and varied scales
including signed zero. Matrix integration tests cover dequantization, dynamic
and weight-only linear operations, Scalar/Auto, one/three input rows, idempotent
packing, unchanged storage, and nonpackable dimensions. The integration RED
initially also has an ambiguous integer type; that test defect is fixed before
the retained missing-method failure (`integration-red-fixed.log`).

The isolated dot median is 23–30% lower across expert, projection and vocabulary
head shapes. Raw results are in `micro-actual.log` and `summary.json`. The first
micro command accidentally uses an incomplete exact test name and runs zero
tests; `micro.log` is retained but supplies **no** performance evidence.

The first isolated unpack loop suggested a speedup, but it did not accurately
represent the existing `Int8Matrix::dequantize` function. A later measurement of
the actual functions, including allocation, shows packed dequantization **10–15%
slower** on representative matrix sizes. Replacing interleaved loads with
explicit SIMD transpose instructions does not fix this: packed remains about
13–14% slower in that follow-up. The actual-function data supersede the simplified
unpack microbenchmark for judging the runtime tradeoff.

## Full-decoder measurements

All runs use AC power, normal power mode, eight workers/BLAS/OMP, a 1517-token
prefix, and 512 teacher-forced decode steps including all-logit hashing. Each
series runs baseline/candidate/candidate/baseline, one warmup and three measured
repetitions per process: six measured observations per variant. All twelve
processes exit normally without forced kills. All **48** repetitions across
three series have the same complete logit SHA-256:
`3e4fa2e2d048750fba24ad7cde61ccc7c92e1bc2c7436e0d387d867387c6c7da`.

| Variant | Decode median, baseline → candidate (s) | Prefill median, baseline → candidate (s) |
| --- | ---: | ---: |
| Initial packed weight-only fallback | 3.333825 → 3.287093 | 1.478006 → 1.609897 |
| Four-output weight-only fallback | 3.328445 → 3.258940 | 1.492318 → 1.540629 |
| Body only, original vocabulary head | 3.328654 → 3.250746 | 1.480321 → 1.535779 |

The initial candidate has a measured 4.347689-second outlier, retained in the
report. The four-output fallback removes redundant four-row reloads and improves
the tradeoff but does not remove it. The final body-only variant reduces decode
median by 2.34% with nonoverlapping measured ranges, but increases prefill median
by 55.46 ms (3.75%). Keeping the original vocabulary head does **not** resolve the
prefill problem; that hypothesis is contradicted by this measurement. The decode
saving over these 512 steps is 77.91 ms, and cannot be assumed to compensate for
prefill on shorter outputs. No full-page speed or OCR accuracy claim is made.

These are shared-host resident measurements. Warmups are excluded by the fixed
protocol; all measured samples remain. Process lifetime includes model loading,
packing, fixture setup and all repetitions, and is not an isolated load benchmark.

## Retained implementation and reproduction

Baseline production source is `c435f58`, using the accepted QKV implementation;
the preserved baseline unit binary identity is in `baseline.json`. Every ABBA
series stores source and executable hashes, commands, power checks, raw timings
and process resources. The accepted CLI remains SHA-256
`1f1902e4db8a0c9ad5ad79242a439c589ba423ab8462cf52754e8af285c1f193`.

`final-experiment.patch` contains the complete experimental source including the
final transpose follow-up and test/microbenchmark harness. It can be applied to
the baseline production tree. `lane_q8-before-transpose.rs.snapshot` retains the
kernel used by the body-only ABBA series; replacing the patched kernel file with
that snapshot reconstructs that algorithm. Earlier tracked-source diffs and
kernel snapshots are diagnostic history, not standalone complete patches.
The transpose follow-up has correctness and actual-dequantization measurements,
but no full-decoder ABBA series.

The preserved ignored unit probes are:

- `cpu::lane_probe::lane_packed_microbenchmark`
- `cpu::quantized::lane_layout_tests::packed_q8_dequantization_probe`

Build the experimental release unit executable, preserve it as
`artifacts/lane-packed-probe/candidate`, and invoke `run.py --output-name FRESH`
with `PYTHONPATH=.` to repeat fixed-work comparison against the pinned baseline.
A new result must retain its actual binary and source identities. Runtime source
was restored after the experiment; `restored-tests.log` verifies the restored
library tests. No accepted benchmark claim is increased by this experiment.
