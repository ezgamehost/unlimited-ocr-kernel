# Paired Q8 block scheduling probe

Hypothesis: computing two adjacent 32-element dot-product blocks before their ordered FP32 additions exposes independent instructions and reduces loop overhead. The four row-major weight streams, quantization, scales and original accumulation order remain unchanged. There is no additional model storage. Odd block counts retain a final single block.

Acceptance requires exact focused tests, identical full fixed-work logit hashes, a repeatable timing improvement, then complete OCR and reference validation. Battery timing is exploratory and requires AC confirmation before a performance claim. The preserved baseline executable has the accepted production source; identity is retained in `baseline.json`.

## Result: rejected

All 16 full-logit hashes match. Baseline median decode including checksum was 3.703801229 seconds; candidate was 3.707072021 seconds, 0.09% slower with overlapping ranges. Prefill medians were 1.473843375 and 1.497927292 seconds. The paired-block candidate provides no useful measured gain and has been reverted. No full OCR acceptance run was warranted. Three focused NEON tests passed; the rejected patch and executable identity are retained with raw ABBA logs.
