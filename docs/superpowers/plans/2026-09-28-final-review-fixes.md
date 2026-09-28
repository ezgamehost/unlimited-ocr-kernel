# Final source-review corrections

Reviewed range: `5d6923b..2e93b0235363a1b61589e5f6eb656bead38dba67`.
The independent read-only review reported no critical defects and three
important defects. Source inspection confirmed all three. Implementation and
test execution must wait until the current Q8 resident ABBA process exits;
do not change its executable or inference source during timing.

## 1. Fresh-process timeout cleanup

`tools/benchmark.py:run_llama` currently kills only its `/usr/bin/time`
wrapper on timeout. Start a dedicated process session, retain its handle,
terminate the whole process group, escalate if necessary, and reap the
wrapper before returning. Preserve partial logs and the timed-out error
classification. Cleanup must also cover interruption/exception paths.

Add a real lightweight regression with a child and grandchild, including
a process that ignores SIGTERM. Verify both stop executing before the helper
returns, and ensure the test itself cleans up on failure. Keep existing
finish-boundary, resource parsing and child-environment coverage. Replace
the mocked subprocess.run parsing test with a real small log emitter if
needed; avoid coupling tests to the new Popen implementation.

This is a defect in the pending fresh-process matrix runner, not evidence
that the resident ABBA series is contaminated.

## 2. Decoder/vision width contract

The generic decoder accepts widths other than 1280, while the vision
projector is fixed at 1280. Reject incompatible width at model loading,
before checkpoint materialization. Keep decoder-only generic configurations.
Validate the feature tensor's exact row count and width before scattering
image features into embeddings; return a structured error rather than
silently regrouping rows or panicking.

Add smaller/larger width rejection tests (640 and 2560), proving rejection
occurs before checkpoint access. Add malformed feature shape coverage for
the scatter boundary, with valid replacement preserving surrounding tokens.

## 3. Convolution temporary allocation

`conv2d` checks patches plus output against the scratch budget, but clones
all weights into a flattened Tensor afterward. Prefer borrowing the weight
slice through a private shared linear implementation, preserving existing
shape validation, scalar accumulation order, BLAS dispatch, and bias order.
Avoid a second kernel implementation solely for convolution.

Verify exact budget boundaries for patches plus output, and numerical
parity on multiple channels, output channels, padding, stride and bias for
Scalar and Auto. If borrowing requires an overly broad change, checked
accounting of the clone is an acceptable correctness fix, explicitly
retaining its performance cost.

## Verification and remaining gates

For each fix, demonstrate a failing regression before the correction, then
pass focused tests. Run the required Rust/Python suites, reference checks,
format and Clippy after the combined fix pass. Build the final CLI only
after tests (integration tests can replace the top-level CLI), preserve its
SHA, then run the relevant exact-output gate before using it in benchmarks.
Keep the just-completed comparison tied to its original executable hash.

The review also identified uncompleted validation, not additional confirmed
defects: exported teacher-forced logits/router fixtures are not all consumed
by tests; broader precision/fresh-process comparison and original-upstream
max32 attribution remain open. The hybrid max32 accuracy failure remains
disclosed. Neither passing these fixes nor the Q8 comparison alone proves
the whole project complete.
