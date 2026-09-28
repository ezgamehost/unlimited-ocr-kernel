# INT4 weight-level accuracy simulation

The candidate passed this limited accuracy screen: all eight synthetic pages
reached EOS with zero normalized character error using the original native
prompt. The Q8 baseline had zero CER on seven pages and one purchase-order
character error. No production runtime or default weights were changed.

**There is no packed INT4 kernel or storage saving in this experiment.** The
candidate uses signed weight levels -7..7 and block32 FP32 scales, physically
stored as I8 and executed by the existing W8A8 runtime. Routers, embeddings,
norms and vision retain original precision; activations use the existing dynamic
INT8 path. The runtime's precision string describes that I8 execution path; the
overlay metadata and this report additionally identify simulated 4-bit weights.

| Page | Q8 CER | Simulated Q4 CER | Q8 / Q4 emitted tokens |
| --- | ---: | ---: | ---: |
| Original English | 0 | 0 | 250 / 250 |
| Original Chinese | 0 | 0 | 289 / 289 |
| Original table | 0.003984064 | 0 | 248 / 247 |
| Original dense | 0 | 0 | 1649 / 768 |
| Frozen receipt | 0 | 0 | 272 / 272 |
| Frozen wide report | 0 | 0 | 355 / 355 |
| Frozen bilingual notice | 0 | 0 | 384 / 384 |
| Frozen archive register | 0 | 0 | 1790 / 1790 |

Both processes exited 0 without forced termination. Each used the original
prompt `<image>document parsing.`, cap8192, ngram35/window128, eight workers and
eight BLAS threads. Each request had independent KV state. The dense page's
formatting/token length changes substantially despite identical normalized
text. Timing differences from these observations cannot establish a kernel
speedup. Raw layout/token differences are retained in the output files.

The exporter test checks literal rounding results (including ties), a zero tail
block, physical I8 dtype and explicit simulation metadata. The full Python suite
passed: 33 tests run, 31 passed and two skipped. The pre-change regression failed
because the four-bit export option did not exist. Default eight-bit export
encoding and metadata are preserved.

`overlay-audit.json` verifies all 2,197 matrices / 2,768,322,560 weight codes are
within -7..7 and records the checksum. The 3,114,914,040-byte candidate is pinned
as SHA256 `693e0503d1faa1ec5e1b4fa2ff6d97c583cce9dd4682a3342cca9d75e8cc4071`.
The original source checksum remains unchanged and both source/output integrity
checks stay enabled. The overlay and sidecar are under
`artifacts/unlimited-ocr-q4-sim-in-q8.*`.

This result is sufficient to investigate packed kernels, not to declare broad
model-quality parity. The pages are synthetic, one inference per configuration,
and now inform this precision experiment. They must not be described as unseen
validation for a future Q4 implementation. Packed kernels must match these
simulated numerical operations before performance testing, and subsequent
quality evaluation must retain additional failures or changes.

Reproduce from the repository root, using new artifact/result paths when needed:

```sh
.venv/bin/python -m tools.quantize --simulate-int4 --output artifacts/unlimited-ocr-q4-sim-in-q8.safetensors
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-int4-accuracy/audit_overlay.py
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-int4-accuracy/quality.py
```

Existing outputs are protected against overwrite. `comparison.json` records CER
deltas, completion, lengths and exact raw-output equality. Full responses,
commands, binary hash, input hashes and overlay manifests are retained for both
configurations. See [PLAN.md](PLAN.md) for the accuracy-to-performance gates.
