# Packed INT4 preparation and representation audit

The packed overlay is 1,730,753,240 bytes versus 3,114,914,040 bytes for the
unpacked INT4 simulation: a 44.4366% file-size reduction including FP32 scales
and headers. This is not a runtime speed or peak-memory result. Packed matrix
kernels, loader dispatch and full inference acceptance are still pending.

Format `uocr-q4-block32-v1` uses U8 tensors shaped `[rows, ceil(cols/32), 16]` and
F32 `.q4_scales` tensors shaped `[rows, ceil(cols/32)]`. Each block byte j stores
weight j in the low nibble and weight j+16 in the high nibble. Codes use signed
two's-complement -7..7; tail positions are zero padded. The source descriptor
supplies the original column count. See the
[implementation plan](../../superpowers/plans/2026-09-28-packed-int4.md).

The independent audit unpacked every one of 2,768,322,560 weight codes across
2,197 matrices. All codes equal the successful accuracy simulation and every
FP32 scale is bitwise identical. Padding and code-range checks passed. Both
source/output checksums remain mandatory. Packed artifact SHA256:
`c98c6134d7b3925b4b7a10bca543afded92cf6b335143d1d17c8acd31e3dfd74`.
`audit.json` retains the full manifest and oracle hash.

The exporter regression uses hand-derived bytes (including 0x79 and 0xE3),
multiple row chunks and an odd-width tail. Full Python tests: 34 run, 32 passed,
two skipped. The new Rust checkpoint regression checks unsigned conversion,
truncation rejection and rejection by ordinary floating-point linear dispatch.
The full Rust suite and clippy passed; logs are retained here. No packed INT4
execution path is enabled by these checkpoint changes alone.

Reproduce preparation/audit from the repository root, using a new output path
when repeating the exporter:

```sh
.venv/bin/python -m tools.quantize --int4 --output artifacts/unlimited-ocr-q4-block32.safetensors
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-packed-int4/audit.py
```

`--int4` and `--simulate-int4` are mutually exclusive. Normal Q8 export retains
its existing format and metadata. The model overlay loader now accepts the packed version after the integration
checks below.

Portable matrix implementation is now available in `src/cpu/int4.rs`:
validated nibble storage, weight-only linear, prefill dequantization and W4A8
single-token arithmetic. Tests cover literal signed bytes, invalid codes and
padding, malformed/overflowing dimensions, scales, row tails and scratch limits.
Across twelve widths (1 through 1280), packed dequantization and dynamic output
match the I8 simulation bit-for-bit on Scalar and Auto. Weight-only outputs
match within floating-point tolerance. A signed-zero regression exposed the
existing scalar/SDOT accumulation identity difference; the packed path preserves
the selected backend's identity. Full suite: 53 passed, 8 ignored; clippy clean.
See `portable-tests.log` and `portable-clippy.log`. The integration below adds SIMD and model loader dispatch.


The four-row ARM SDOT path unpacks signed low/high halves in registers, shares
activation loads and preserves separate scale products and block accumulation.
Runtime dotprod detection guards entry; row and column tails retain the portable
path. Q4 loading checks format-specific metadata, original dimensions, U8 packed
shape, FP32 scales, code range, padding, source hash and overlay hash. Q8 loading
and original BF16 default remain supported. `--quantized-weights` aliases the
existing `--int8-weights` option; use `--int8-activations` for W4A8 execution.
Precision reports identify packed Q4 and dynamic INT8 activations explicitly.

Integration checks: 56 Rust tests passed, 8 artifact-dependent tests ignored by
the default suite; clippy clean. Separately, all ten decoder/vision tests ran
with ignored original-reference cases enabled and passed. See the integration
and original-reference logs. This is separate from the final whole-branch review.

`equivalence.py` ran all eight selection pages with the original prompt. Every
text string and token ID matches the earlier unpacked Q4 simulation exactly;
all eight reached EOS. Raw requests, responses, executable hash and normal-exit
resources are retained under `packed-equivalence/`. The process used a measured
peak RSS of 5,117,886,464 bytes, but this single accuracy run is not a repeated
latency or memory comparison. These synthetic pages informed precision selection
and are not unseen Q4 validation.

Initial fixed-work ABBA comparison (`fixed-work/`): six measured samples per
variant, plus four warmups total. All sixteen full-logit SHA256 values match
`abf6835c8398b0f760f50a31f053e4633656b6fa9675d5b6305317ecc384ed3f`.
Median 512-step decode including checksum improves from 3.845986 to 3.205082 s
(1.200x throughput), while prefill regresses from 1.468385 to 2.735561 s.
Packed process peaks are 3,491,840,000 and 3,491,823,616 bytes; raw simulation
peaks and all per-process resources are retained. This initial candidate is
not accepted as an overall speed improvement: its prefill regression needs
correction. These compare the same Q4 numerical model in two storage formats,
not Q8 quality or llama.cpp.
