# Unlimited-OCR CPU runtime

Native Rust inference for Baidu Unlimited-OCR, currently optimized and measured
on an Apple M4 Pro. The runtime implements image preparation, SAM/CLIP vision,
MoE decoding, preserved-prefix KV caching and generation. It uses custom NEON
kernels and Apple Accelerate math primitives. PyTorch is used only for offline
reference generation and optional weight preparation; llama.cpp is a benchmark
competitor, not a runtime dependency.

This is an active performance project. Portable scalar fallbacks exist, but
other CPU platforms have not received equivalent optimization or benchmarking.
On the four primary synthetic pages, the latest repeated resident comparison
measures a **1.30–1.54× median full-page speedup over llama.cpp Q8** with the
native dynamic-activation Q8 decoder. Standalone token decoding remains slower.
See the [Q8 full-page results and limits](docs/benchmarks/2026-09-28-current-q8/README.md).
The hybrid has a confirmed accuracy regression on the
[32-crop stress input](docs/benchmarks/2026-09-28-hybrid-max32/README.md); the
four-page speedup does not establish general quantization accuracy.

## Build and run

The toolchain is pinned in `rust-toolchain.toml`. On macOS, building also requires
the command-line developer tools for native dependencies and Accelerate linking.

```sh
cargo build --release
python3 -m tools.fetch_model
target/release/uocr --image benchmarks/corpus/english.png --threads 8 --json
```

The download helper uses Python's standard library and pins the original model
revision `07dea832e22aefee32ad281d4b80551282e1c168`. Weights and their manifest
live in `artifacts/upstream`; they are not checked into Git. Downloaded model
Python files are not executed by the Rust runtime. The upstream model license
is downloaded alongside the weights.

PNG and JPEG inputs are supported. Default `--mode gundam` uses a global view
and adaptive local crops; `--mode base` uses the global view. The default prompt
is `<image>document parsing.`. Use `--max-tokens` to bound generation; a
`token_limit` finish is incomplete and must not be counted as a completed OCR
benchmark. `--help` lists the full CLI.

`--threads` sizes the Rust worker pool. On macOS 15+, parallel decode attention
uses single-threaded BLAS calls to avoid nested workers; older systems use
serial attention heads. Dense vision/prefill matrix multiplication still uses
Accelerate's automatic threading, so this option is not a global thread cap.

## Keep a model resident

```sh
target/release/uocr --jsonl --threads 8
```

Send one request per line on stdin:

```json
{"image":"benchmarks/corpus/english.png","generation":{"max_tokens":2048}}
```

Each request starts an independent generation cache. Responses contain either
`status: "ok"` and a result, or `status: "error"` and an explanation. Results
include raw text, token IDs, finish reason, precision, image-token count and
phase timings. Load time is emitted separately on stderr. Resident inference
timings exclude load; fresh-process comparisons must include it.

The resident benchmark runner accepts `expected_image_tokens` on each manifest
image. It defaults to 1513 for the original four-page corpus. For other crop
layouts, declare the independently expected count; native responses and llama
prompt lengths are checked against it. The fixed matched prompt adds five
non-image tokens. Per-image expectations are saved in process metadata; a mixed
manifest has no single `prefix_tokens` value. See the
[real-runtime adapter check](docs/benchmarks/2026-09-28-variable-token-runner/README.md).

## Optional INT8 decoder

Prepare an overlay in the offline Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reference.txt
.venv/bin/python -m tools.quantize --output artifacts/unlimited-ocr-q8-block32.safetensors
target/release/uocr --image benchmarks/corpus/english.png --threads 8 --json \
  --int8-weights artifacts/unlimited-ocr-q8-block32.safetensors
```

This keeps FP32 activations. Adding `--int8-activations` enables the separately
labeled experimental mode that also rounds single-row projection inputs,
including single-row expert groups during prefill. Routing, normalization,
embedding and vision weights remain original in both modes. The original
checkpoint is still required. The loader verifies complete source and overlay
checksums, version, shapes, dtypes, ranges and scales before dispatch.

BF16 remains the default. Both INT8 modes introduced one table character error
in the four-page pilot where native BF16 had none. See the retained
[INT8 format and preparation](docs/benchmarks/2026-09-28-int8-kernels/README.md)
and [page results](docs/benchmarks/2026-09-28-parallel-activations/README.md).

## Experimental packed INT4 decoder

Keep the output head at Q8 to avoid the observed all-Q4 table regression:

```sh
.venv/bin/python -m tools.quantize --int4-q8-head --output artifacts/unlimited-ocr-q4-q8head-block32.safetensors
target/release/uocr --image benchmarks/corpus/english.png --threads 8 --json \
  --quantized-weights artifacts/unlimited-ocr-q4-q8head-block32.safetensors --int8-activations
```

The decoder body uses signed packed block32 weights with FP32 scales and ARM
SDOT under runtime feature detection, plus a portable fallback. The original
checkpoint is still required for unquantized tensors. This hybrid matches its
simulation exactly on twelve prompt/page cases, all at zero character error.
The accepted vision optimizations preserve those exact outputs. A 96-request
resident comparison is complete: Rust median page times are 9.03–9.32s on the
shorter pages and 12.49s on dense, versus llama's 12.94–13.26s and 15.69s. Rust
has zero character errors on all four pages; llama has one table character
error. Peak process RSS is 5.05–5.06GB versus 9.29–9.80GB. These systems use
different precision and policies; broader quality and fresh-process validation
remain pending. See [full-page evidence](docs/benchmarks/2026-09-28-packed-int4-resident/powered-abba-2/README.md)
and [hybrid format evidence](docs/benchmarks/2026-09-28-hybrid-int4/README.md).

On the synthetic 32-crop input, hybrid Q4/Q8 with dynamic INT8 activations
produces unrelated repeated text: character error rate 1.3664, versus 0.0691
for current BF16 and dynamic Q8. Both cache layouts reproduce the hybrid
failure exactly. BF16 and Q8 retain their earlier outputs, so this observed
regression is specific to the hybrid precision path. The default runtime
continues to use original BF16 weights. See the
[precision controls and raw outputs](docs/benchmarks/2026-09-28-hybrid-max32/README.md).

For the measured eight-worker setup, set `VECLIB_MAXIMUM_THREADS=8` and
`OMP_NUM_THREADS=8` in the environment before starting the process, alongside
`--threads 8`. CLI worker count alone does not cap Accelerate's threads.

All-Q4 export with `--int4` remains available. It passes the eight original-prompt
pages but returns an empty table with the extra newline used in the llama.cpp
comparison. See the [precision diagnostic](docs/benchmarks/2026-09-28-packed-int4-resident/README.md).

## Rust API and checks

`Model::load` loads the original checkpoint; `Model::load_with_quantization`
selects an overlay and explicit `QuantizationMode`. Create a `CpuContext` with
backend, worker count and scratch budget, then call `Model::infer(&Request,
&mut CpuContext)`. The scratch limit applies to checked operator allocations;
it is not a total process-memory limit. Mapped checkpoint contents must remain
immutable throughout their lifetime.

Manually constructed `PreparedImage` values are validated before vision inference.
`PreparedImage::image_tokens()` returns `Result<usize>` and rejects inconsistent
crop grids, unsupported view sizes, and malformed tensor storage.
`Decoder::new` also validates directly constructed model configurations;
`Decoder::config()` provides read-only access after construction.

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

Original-weight decoder and vision tests are ignored by default and require
the downloaded weights and offline reference fixtures. With those present:

```sh
cargo test --release --test decoder -- --ignored --test-threads=1
cargo test --release --test vision -- --ignored --test-threads=1
```

The [implementation plan](docs/superpowers/plans/2026-09-27-rust-unlimited-ocr.md)
and [design](docs/superpowers/specs/2026-09-27-rust-unlimited-ocr-design.md) describe
the remaining scope. Benchmark reports retain raw output and disclose differing
precision, attention policies, generation lengths and process lifetimes. The
[repeated resident INT8 comparison](docs/benchmarks/2026-09-28-resident-repeated/README.md)
completed 40 measured requests per engine: Rust's median page speed was
1.08–1.28× llama.cpp's, with matching text CER and 26% lower peak child RSS.
Rust still had slower decode throughput and slightly worse dense-page p95.
On the [frozen-page repeated check](docs/benchmarks/2026-09-28-heldout-repeated/README.md),
the two pages accurate in both engines showed mixed, small timing differences.
Native transcribed the archive accurately while llama repeated/reordered text.
The [llama BF16 baseline](docs/benchmarks/2026-09-28-bf16-baseline/README.md)
is also complete, with host paging documented. Full precision/fresh-process
acceptance and final review remain unfinished.

For the pinned llama.cpp comparison, use an explicit newline after `<image>`:
its image processor inserts that newline automatically. The [matched-prompt
pilot](docs/benchmarks/2026-09-28-thread-controls/README.md) records the exact
thread settings, raw outputs and why the earlier dense-page token counts differed.
This comparison setting does not change the runtime's default prompt.
The newline is not quality-neutral: it makes the frozen receipt return empty
text, while the unchanged native default recognizes it with zero CER in both
BF16 and INT8. The [original-model diagnostic](docs/benchmarks/2026-09-28-receipt-diagnostic/README.md)
reproduces that first-token distinction in FP32 and BF16. Use the original
default prompt for normal inference; retain the newline protocol only when
explicitly measuring that matched-prefix configuration.
