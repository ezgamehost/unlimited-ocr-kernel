# Upstream FP32 prefill thread sensitivity

Numerical diagnostic only, run on battery; no latency measurements or production
changes. The process exited successfully. Reproduce from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-reference-thread-sensitivity/probe.py
```

The probe refuses to overwrite its artifacts or report. Move prior output aside
before reproducing. It reuses the frozen logo image embeddings and prompt, calls
the hash-verified original upstream decoder forwards with BF16 weights expanded
to FP32, and runs prefill at ten threads followed by one thread. No vision rerun
or native implementation is involved. Source and fixture hashes are in results.json.

| Comparison | Maximum absolute logit difference | RMS difference | Current tolerance violations |
|---|---:|---:|---:|
| Ten threads vs original fixture | 0.0000224113 | 0.00000264064 | 0 |
| One thread vs original fixture | 0.0001226664 | 0.0000263779 | 0 |
| One thread vs ten threads | 0.0001211166 | 0.0000262585 | 0 |

The tolerance shown is `1e-4 + 1e-4 * abs(reference)`, applied to all 129,280
prefill logits. Every decoder hidden-state tensor at ten threads is bitwise equal
to the original fixture's prefill slice. The original output projection has nine
rows while this prefill-only probe has one, so that comparison also changes the
matrix shape at the final head. The one-versus-ten comparison holds that shape
fixed. Its final-layer hidden-state maximum difference is 0.0000648499.

Sorting selected expert ID/weight pairs by ID confirms all 18,282 prefill route
IDs match between thread counts across all eleven MoE layers; routing.json records
per-layer weight differences. Thus this experiment shows rounding sensitivity
without a discrete routing change.

This is evidence that the FP32 reference is sensitive to execution configuration,
not proof that native BF16 parity passes. Native's previously recorded 24 prefill
logit violations remain open. No threshold has changed. A higher-precision oracle
and direct native comparison remain necessary before calibrating a full-model
tolerance; simply converting the upstream modules to FP64 is insufficient because
several upstream operations explicitly force FP32.
