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
its existing format and metadata. The current model overlay loader still
rejects the packed version until its planned integration is implemented.
