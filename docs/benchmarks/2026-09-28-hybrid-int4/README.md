# Packed Q4 decoder with Q8 output head

The mixed runtime retains Q8 precision only for `lm_head.weight`. Other eligible
decoder matrices use the existing packed Q4 representation. This addresses the
all-Q4 table failure under the matched-newline prompt; the numerical choice was
first checked using the [I8 simulation](../2026-09-28-packed-int4-resident/README.md).
BF16 default, Q8 and all-Q4 remain supported. No new arithmetic kernel is needed.

Format `uocr-q4-q8head-block32-v1` requires body_bits=4 and head_bits=8 in both
header and checksum manifest. The head is I8 `[rows,cols]` with `.q8_scales`;
other eligible matrices are U8 `[rows,ceil(cols/32),16]` with `.q4_scales`.
The loader requires the head, exact matrix sets, format-specific dtypes/shapes,
valid scale/code/padding values and verified source/output checksums. There is
no arbitrary mixture of precisions within this version.

```sh
.venv/bin/python -m tools.quantize --int4-q8-head \
  --output artifacts/unlimited-ocr-q4-q8head-block32.safetensors
target/release/uocr --image benchmarks/corpus/table.png --threads 8 --json \
  --quantized-weights artifacts/unlimited-ocr-q4-q8head-block32.safetensors --int8-activations
```

The exporter flag is mutually exclusive with `--int4` and `--simulate-int4`.
Runtime precision is reported as `int4_body_int8_head_dynamic_int8_activations`
or `int4_body_int8_head_fp32`, depending on activation mode.

The real artifact is 1,813,492,888 bytes, SHA256
`6981387fed4849dca9a713a717e9b54c04344adbaeff05410528773912e287ae`.
Its independent audit matches every one of 2,768,322,560 codes across 2,197
matrices, and every FP32 scale bit, to the hybrid simulation. It is 41.7803%
smaller than that physical-I8 artifact; file size does not establish speed or
peak process memory.

Tests: Python 35 run, 33 passed, 2 skipped; Rust 57 passed, 9 ignored; clippy
clean. Separately all ten decoder/vision tests passed with original references
enabled. Tests cover literal mixed-format bytes, tails, source preservation,
invalid exporter modes, missing head, metadata, wrong dtypes/suffixes/ranges,
padding and accurate precision labels. Default Q8/all-Q4 regressions still pass.

All twelve full OCR outputs match the hybrid simulation's text and token IDs
exactly: eight original-prompt pages plus four matched-prompt pages. All reach
EOS, inheriting the simulation's CER0 on these cases. Raw outputs, requests and
binary/process identity are in `packed-equivalence/`. Original dense emits
1578 tokens while matched dense emits 768; do not hide that difference when
interpreting timing. These synthetic pages informed precision selection and
are not unseen validation. Repeated full-page acceptance remains open.

Fixed-work ABBA comparison (`fixed-work/`): six measured repetitions per variant,
plus warmups; all sixteen full-logit SHA256 values match
`c129d845d9d0f6b2b1f7bc1aaa7741d1a9fb95b428a1016f519df94acb862f3a`.
Median prefill is 1.607606 s packed versus 1.471638 s simulated; 512 decode steps
including checksums are 3.405770 s versus 3.868158 s. Decode throughput improves
13.58%, while prefill is 9.24% slower. These are the same hybrid numerical model;
they do not quantify performance against llama.cpp. Raw process peaks and all
samples are retained, including variability between the two packed processes.
