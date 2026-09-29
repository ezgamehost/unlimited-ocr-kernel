# Pairwise FP32 RMS reduction: global candidate not accepted

The prototype recursively sums squared inputs in balanced halves, with serial
leaves of at most sixteen values. It changes only the uncentered RMS variance
reduction. The complete implementation/test patch is candidate.patch; production
source was restored after evaluation.

The targeted wide-row test failed on the accepted scalar reduction: width 1279,
one value 1 and all others 1e-4, produced 35.74026 versus FP64 35.7400328049 for
the large normalized value. The candidate passes all 18 operator tests, including
the new FP64-relative-error bound across widths 1279/1280/1281 and three positions
of the large value.

All nine original full-model teacher-forced logit checkpoints pass unchanged
tolerance, covering prefill and 300 steps through two cache wraps. Prefill maximum
error is 0.0001134872, RMS 0.0000234568, zero violations (accepted runtime: maximum
0.0001540184, RMS 0.0000400269, 24 violations). Top-1 agrees in every row. This is
a candidate result; the accepted runtime's open parity gate is not closed.

The twelve-page Q8 process exits normally, without a forced kill, and every page
reaches EOS. All normalized text and token lengths remain equal, but seven pages
have changed coordinates (maximum coordinate delta three). The exact-output
quality harness consequently exits 1. It stops before hybrid; hybrid was not run.

Independent scoring against frozen rendering bounds gives mixed layout results:
English matched, Chinese original, and receipt improve mean IoU; Chinese matched,
dense original, and archive register worsen; bilingual notice ties. Thus a global
replacement does not preserve the existing output guarantee and has no uniform
layout benefit. No performance claim is made: the machine was on battery and this
was a correctness/quality experiment. The candidate CLI is retained by the quality
harness with source hashes in quality/candidate-cli.json.

The coordinate and layout JSON reports use the existing teacher-logits scripts
with ROOT/CANDIDATE/NEW redirected to this directory and quality/q8; their original
baseline paths remain unchanged. The original scripts are in
../2026-09-28-teacher-logits/{coordinate_diff,layout_score}.py. They assert text
alignment and validate the frozen image/font provenance before scoring.

No global change is accepted. A BF16-specific numerical improvement may still be
useful, but needs its own broad BF16 OCR comparison and performance assessment;
these Q8 results cannot establish that. The global candidate remains available
for that investigation without altering the accepted benchmark runtime.
