# Shared Q/K/V and MoE activation quantization

Accepted implementation: a prepared activation owns the quantized row and its
rank, with no copied FP32 input or extra shape allocation. Concurrent dynamic
Q/K/V projections share one preparation instead of three. Parallel MoE execution
shares one preparation across the gate/up projections of six routed experts and
the shared expert, instead of fourteen. Each down projection still quantizes its
own distinct intermediate. Prepared data is local to the invocation and immutable.

The existing aggregate scratch gates, per-projection width/resource validation,
and serial/low-scratch/multirow/unquantized/weight-only fallbacks remain intact.
Q8 and packed Q4 use the same prepared-row interface; arithmetic and accumulation
order are unchanged. No additional model storage is introduced.

## Verification

- Final standard suite: 74 passed, 17 opt-in tests ignored; fmt and clippy pass.
- Independent expert/decoder references pass.
- The new opt-in test checks bit-identical prepared versus independent projection
  results for dense, shared and routed experts in Q8 and hybrid modes. It also
  checks complete layer outputs with one worker, eight workers, and eight workers
  constrained to 32 KiB scratch, covering the parallel and fallback branches.
- All four independent full Q8 decoder-logit digests match the accepted runtime.
- All twelve Q8 and twelve hybrid OCR outputs match exactly: raw text/coordinates,
  token IDs, EOS, image-token counts and precision. Both processes exit normally.
- Static reviews prepared_activation_review and moe_prepared_review report no
  blocking findings. They examined rank reconstruction, resource accounting,
  immutable parallel borrowing, invocation lifetime, and fallback behavior.

The missing-method RED and successful targeted tests are retained. The reviewer
noted that prepared/ordinary operator tests share implementation, so the independent
model-digest and OCR checks above remain essential. Final CLI SHA256 is
`c05f6682b64b8d0326c1f859003dfdd66c29a1c063cc59e59e56882bf34de835`, identical
to the executable used for OCR validation. The final test/build and OCR sessions
all completed; no validation job remains running.

## Battery-only performance observation

The comparison baseline is the accepted borrowed-activation implementation
(ce84f3f), before both sharing changes. ABBA order uses four processes, each with
one predeclared warmup and three measured repetitions. All sixteen complete logit
hashes match. Every child exits normally without a forced kill. Battery source,
normal power mode and charge of at least 50% pass all before/after checks.

| Six measured samples per variant | Baseline | Shared Q/K/V + MoE |
|---|---:|---:|
| Decode median seconds | 3.629515 | 3.450077 |
| Decode min–max seconds | 3.408028–4.106251 | 3.305436–3.841180 |
| Prefill median seconds | 1.571845 | 1.474869 |
| Prefill min–max seconds | 1.443293–1.990742 | 1.379217–1.747591 |

The observed decode median is 4.94% lower. Ranges overlap, and this is a battery
screen, so it is not a confirmed AC speedup or a new whole-page comparison with
llama.cpp. All outliers remain in the summaries. The Q/K/V-only experiment in the
adjacent shared-activation directory showed no established gain; this report
evaluates the combined implementation. The accepted rationale is elimination of
repeated work with unchanged validated outputs and no established regression.

run.py reproduces the fixed-work comparison with --power-source and a fresh
--output-name. validate.py records standard/reference validation. The additional
prepared_expert_inputs_preserve_independent_projection_outputs opt-in library test
records fallback coverage. The existing final-review/quality.py harness checks
both quantized OCR modes. These changes do not resolve the known hybrid 32-crop
accuracy failure or establish standalone decode superiority over llama.cpp.
