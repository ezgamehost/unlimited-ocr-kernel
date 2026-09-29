# Upstream expert-routing reference gates

Two new opt-in tests consume the previously unchecked expert-ID and expert-weight
fixtures for all eleven MoE layers. **All 38,082 selected expert IDs match exactly
in each test**, covering 277 prefill positions and 300 teacher-forced positions
per layer. This closes fixture coverage for routing; it does not establish
whole-model numerical parity or general OCR accuracy.

| Gate | Input propagation | Largest routing-weight absolute difference |
| --- | --- | ---: |
| Layer-isolated | Previous-layer hidden states supplied by the upstream trace | 2.5033951e-6 |
| Full native trajectory | Reference input embeddings, then native prefill and 300 native decode steps | 1.0073185e-5 |

Each test compares 11 × 577 × 6 selected IDs and corresponding weights. It sorts
ID/weight pairs by ID because upstream exports `topk(sorted=False)` ordering;
sorting never separates weights from their expert IDs. Expected shape/dtype,
uniqueness, expert range, finite/nonnegative weights, captured row count and
positions are checked. The full-trajectory test has zero ID mismatches and zero
weight violations under the preselected bound `1e-5 + 1e-4 * abs(reference)`.

That bound was set before measurement and is stricter in absolute error than
the existing decoder-layer check. It is a diagnostic routing threshold, **not**
a completed reference-precision calibration for whole-model parity. The existing
24-prefill-logit tolerance failure and general precision calibration remain open;
no failing tolerance was relaxed by this work.

Both tests explicitly pin the model dimensions/routing configuration and verify
head-major cache execution, absolute positions and bounded active counts beyond
two generated-token ring wraparounds. A read-only review identified the need for
those scope assertions; they were added and both final gates rerun successfully.

## Instrumentation and validation

Actual calls to `moe::route` are captured by a `cfg(test)` thread-local hook. The
test starts capture inside `CpuContext::run`, so routing and collection occur
on the same worker even when inner kernels use the shared pool. RAII clears
capture during unwinding; nested capture is rejected. Counts are asserted so
missing callbacks cannot produce a vacuous pass.

The first test fails for the missing capture API (`red.log`), then passes after
instrumentation (`green.log`). The full-trajectory gate independently passes
(`teacher-forced.log`). `final-routes.log` reruns both with the strengthened scope
assertions: two passed, 22 per-layer reports. Standard release tests pass **71**,
with fifteen opt-in tests ignored; formatting, strict clippy, the original expert
scheduling reference and existing decoder-layer reference also pass.

The release CLI is **byte-identical** to the accepted runtime:
`05616a6d17bae60ceda0fefa365dcdb0f93b6987e00246cd8bb8979a7178ee5e`.
The capture is absent from that executable. Unit-test binaries do contain the
inactive capture check, so future fixed-work comparisons should use baseline and
candidate binaries built with this same test harness, rather than silently
mixing harness versions. A preserved current unit baseline is recorded in
`unit-baseline.json`.

This work ran after the laptop switched to battery; it is correctness evidence,
not performance timing. `fixture-identity.json` pins the original model, config,
fixture and export metadata. The fixture uses original BF16 weights converted to
FP32 for upstream CPU computation. Raw reports and final source identities are
retained. Reproduce both gates with the fixtures installed:

```sh
VECLIB_MAXIMUM_THREADS=8 OMP_NUM_THREADS=8 cargo test --release --lib expert_routes_match_upstream -- --ignored --nocapture --test-threads=1
```

`validate.py` additionally runs the standard suite, lint, existing references and
final release build into a fresh validation directory. This report makes no new
speedup claim.
