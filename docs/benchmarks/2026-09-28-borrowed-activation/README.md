# Borrowed dynamic-activation quantization candidate

Status: accepted allocation/scratch reduction; no model-latency improvement claimed.
Standard validation, review and all 24 Q8/hybrid OCR checks pass.

Dynamic Q8 and Q4 projections previously cloned their complete floating-point
activation row into a temporary Tensor before quantization. The candidate factors
the existing quantization loop into a checked borrowed-slice helper and quantizes
the original row directly. Rounding, block order, scales, and output accumulation
remain unchanged. Existing one-row inputs of rank one, two, or three remain valid.
The conservative scratch allowance drops by one FP32 input row in each path.

The new reduced-scratch test failed on the old implementation with Resource
"operation exceeds scratch limit", then passed on the candidate for Q8 and Q4
and all three input ranks. A prior test-construction compile error is retained
separately; it is not counted as the numerical/resource RED result. All INT4 and
quantized integration tests pass. All four fixed-work Q8 full-logit digests equal
the accepted 3e4fa2e2d048750fba24ad7cde61ccc7c92e1bc2c7436e0d387d867387c6c7da.
Those checks are correctness evidence, not speed evidence.

validate.py executes the remaining standard tests, formatting, lint, existing
expert/decoder reference gates and final CLI build, recording each exit status.
All validation steps completed successfully. Independent reviewer
borrowed_activation_review found no issues: scale capacity cannot overflow because
rows * ceil(cols / 32) is bounded by the input element count, the scratch reduction
remains conservative, and single-row shape/borrowing semantics are preserved.
All twelve Q8 and twelve hybrid outputs exactly match the accepted baselines,
including coordinates, token IDs, EOS, image-token count and precision. Both OCR
processes exit normally without forced kills. The tested and preserved CLI is
41da7f1f5241bacbf7464c1a1ba36bff5aca436101cc7b98d84f53efa64d767b.

## Battery ABBA screen

The accepted scoped-RMS unit binary is the baseline; candidate-unit.json identifies
the copy-removal unit binary. Four phases run baseline/candidate/candidate/baseline,
each with one predeclared warmup and three measured repetitions. Every phase
finishes normally; all sixteen complete logit hashes are identical. Battery power,
charge above 50%, and normal power modes are checked before and after each phase.

| Six measured samples per variant | Baseline | Candidate |
|---|---:|---:|
| Decode median seconds | 3.557550 | 3.629836 |
| Decode min–max seconds | 3.196966–4.494192 | 3.409880–4.819886 |
| Prefill median seconds | 1.437751 | 1.521132 |
| Prefill min–max seconds | 1.365156–2.355123 | 1.368853–1.651346 |

The candidate decode median is about 2.03% higher, with broadly overlapping ranges.
No outliers are removed. This battery-only series supports neither a speedup nor
a reliable regression conclusion and does not change the AC benchmark headline.
Acceptance is for removal of one temporary FP32 activation allocation/copy per
dynamic projection and the corresponding lower scratch requirement, backed by
exact operator, decoder-digest and OCR checks. This is not evidence that the
standalone decode gap to llama.cpp has closed.
