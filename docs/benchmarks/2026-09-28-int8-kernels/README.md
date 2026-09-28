# Experimental INT8 kernels and preparation

These kernels are not yet connected to `Model` or the OCR CLI. No quantized
OCR accuracy or whole-page speed claim is established.

The weight-only path retains FP32 activations. The separately named
`linear_dynamic_int8` experiment also rounds single-token activation blocks
to INT8 and uses ARM dot-product instructions when runtime feature detection
permits. It has a scalar integer fallback. This extra activation quantization
needs its own accuracy evaluation before any default dispatch decision.

`cargo run --release --example bench_int8` compares actual expert and vocabulary
matrices: mode 0 is mapped BF16, mode 1 is weight-only INT8, mode 2 is dynamic
activation INT8. The captured run shows vocabulary projection near 1.0–1.3 ms
in mode 2 versus 1.7–1.8 ms in mode 0. Expert-matrix performance did not improve.
These are exploratory hot-matrix samples, not full-model or randomized repeated
benchmarks. The current scheduling/kernel work is insufficient to claim an
INT8 inference speedup.

Prepare the versioned decoder overlay:

```sh
python -m tools.quantize --output artifacts/unlimited-ocr-q8-block32.safetensors
```

The original file remains untouched. Export uses bounded row chunks and an
atomic exclusive publication. It rejects nonfinite and non-floating weights,
verifies the source hash before/after and reopens the finished file with the
safetensors validator. The completed original-model export contains 2197
matrices in 3,114,913,992 bytes. Its source/output hashes are recorded in
`overlay-manifest.json`. Routing, normalization, embedding and vision tensors
are excluded and still require the original checkpoint.

## Format contract: uocr-q8-block32-v1

- Safetensors metadata: `format`, `block_size` (string `32`) and `source_sha256`.
- Each selected decoder matrix keeps its original name and `[rows, cols]` shape,
  with dtype `I8` and values in `[-127,127]`.
- A corresponding `<name>.q8_scales` tensor has dtype `F32` and shape
  `[rows, ceil(cols/32)]`. Scales use little-endian safetensors storage.
- Each row is divided into consecutive 32-column blocks. The final block may
  be short. Scale is `max(abs(block))/127`; quantization rounds ties to even
  and clamps to `[-127,127]`. All-zero blocks use zero scale and zero integers.
- Preparation rejects nonzero blocks whose scale underflows to zero. Runtime
  must check dtype, shape, finite nonnegative scales, version, checksum and
  source binding before dispatch.
- The sidecar `.json` records schema version, format, block size, source/output
  SHA256, matrix count and file byte count.

Tests cover zero ranges, tails, error bounds, scalar and NEON dots against FP64
oracles, invalid scales/storage, parallel row partitions, source preservation,
integer-source rejection, and exclusion of router/normalization tensors. These
are implementation checks, not evidence of model-level quantization quality.
