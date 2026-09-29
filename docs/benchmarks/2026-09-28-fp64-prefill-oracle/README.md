# Independent FP64 prefill precision experiment

The independent FP32 graph matches the upstream prefill fixture bit for bit for
all twelve complete hidden-state tensors, all eleven expert ID/weight tensors,
and all 129,280 logits. Only after that check passes does probe.py execute FP64.
Both runs completed normally. The result favors investigating native numerical
accuracy, not relaxing the existing tolerance.

| Compared with the FP64 graph | Maximum absolute logit error | RMS error | Violations |
|---|---:|---:|---:|
| Original upstream graph, FP32, ten threads | 0.0001086506 | 0.0000219009 | 0 |
| Accepted native BF16-weight runtime | 0.0001720910 | 0.0000442407 | 63 |

Violations use `1e-4 + 1e-4 * abs(FP64 reference)`. This is a diagnostic application
of the existing threshold, not a new calibrated acceptance rule. All 18,282
selected expert IDs agree between FP32 and FP64. Native last-position hidden-state
RMS error exceeds upstream FP32 error at each of the twelve layers.

The FP64 stability rerun uses one thread instead of ten. Maximum logit difference
is **1.279e-13**, RMS **2.529e-14**, with unchanged selected expert sets. This is
many orders of magnitude below the FP32/native differences under investigation.

The graph loads the hash-verified original BF16 checkpoint one projection at a
time. It implements prefill attention, half-split RoPE, RMS normalization, SiLU
MLPs, unnormalized top-six routing, sorted expert grouping, shared experts, final
normalization, and output projection directly in Torch operations. FP64 widens
every floating-point operation, including the normalization, routing, attention
softmax, and rotary operations which original upstream code explicitly forces
to FP32. It is therefore a higher-precision mathematical reference, not a claim
that the upstream software supports this execution mode.

Scope: one 277-position logo prefill, fixed exported FP32 image/input embeddings,
and original checkpoint. No vision reference, decode/cache validation, Q8/Q4
accuracy conclusion, general proof of FP64 exactness, or performance measurement
is implied. The graph is intentionally specialized to this checkpoint. The exact
FP32 validation is its safeguard against a transcription or shape error.

Reproduce from the repository root with the project Python environment:

```sh
.venv/bin/python docs/benchmarks/2026-09-28-fp64-prefill-oracle/probe.py
.venv/bin/python docs/benchmarks/2026-09-28-fp64-prefill-oracle/stability.py
```

Prerequisites are the adjacent reference-thread-sensitivity snapshots and native
snapshot from native-reference-sensitivity. Output artifacts/reports refuse
overwrite; move previous outputs aside to repeat. Logs and JSON reports retain
raw errors and hashes. No production code or tolerance changed. All measurements
here are numerical diagnostics on battery power.
