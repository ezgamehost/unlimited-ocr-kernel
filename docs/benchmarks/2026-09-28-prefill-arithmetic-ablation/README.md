# Prefill arithmetic-policy ablation

Four runs of the independently validated graph isolate RMS and attention-scaling
policies. The unchanged control again matches all upstream FP32 graph outputs
bit for bit. Each variant changes only the indicated policy; no production
runtime code changed. All four runs exited normally with unchanged upstream
expert-selection sets.

| Policy | First-layer last-hidden RMS error vs FP64 | Final-logit RMS error vs FP64 | Final-logit RMS difference vs native |
|---|---:|---:|---:|
| Upstream control | 9.552e-8 | 2.190e-5 | 3.995e-5 |
| Sequential FP32 RMS sum, sqrt then reciprocal | 3.314e-7 | 3.224e-5 | 2.675e-5 |
| Attention multiply by rounded reciprocal sqrt | 9.121e-8 | 1.532e-5 | 4.438e-5 |
| Both changes | 3.331e-7 | 3.921e-5 | 2.067e-5 |

The RMS-policy change increases first-layer error about 3.5 times and moves the
graph toward native output. This supports RMS reduction as a contributor to
native drift. It does not prove RMS is the only source. The intervention includes
both scalar accumulation and sqrt/reciprocal evaluation, so it does not isolate
those two components from each other. The attention-scale intervention improves
FP64 agreement alone on this fixture, while combining it with scalar RMS worsens
final error: floating-point effects do not combine linearly.

Native itself has final-logit RMS error 4.424e-5 versus this FP64 graph. Remaining
differences include matrix-library reduction paths, rotary evaluation, softmax,
expert accumulation order, and activation functions. This experiment does not
establish causality for any of those remaining operations.

The script extracts the validated graph's source, asserts each replacement occurs
exactly once, and saves each executed graph under artifacts/prefill-arithmetic-ablation.
It uses fixed FP32 input embeddings and BF16 checkpoint weights expanded to FP32.
Results retain generated-source hashes. The control's exact equality check guards
against changing the shared graph inadvertently. This is a single prefill fixture,
not a general accuracy result or a speed measurement; the machine was on battery.

Reproduce after the adjacent FP64/reference/native fixture probes:

```sh
.venv/bin/python docs/benchmarks/2026-09-28-prefill-arithmetic-ablation/probe.py
```

The output artifact directory and results report refuse overwrite. No tolerance
changed. A more accurate native RMS reduction is the next candidate to evaluate,
but the earlier FP64-reduction candidate's quantized OCR regressions remain a
reason to require full quality checks before accepting any replacement.
