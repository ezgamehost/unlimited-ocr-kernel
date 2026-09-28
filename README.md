# Unlimited-OCR CPU runtime

Native Rust inference for Baidu Unlimited-OCR, currently optimized and measured
on an Apple M4 Pro. The runtime implements image preparation, SAM/CLIP vision,
MoE decoding, preserved-prefix KV caching and generation. It uses custom NEON
kernels and Apple Accelerate math primitives. PyTorch is used only for offline
reference generation and optional weight preparation; llama.cpp is a benchmark
competitor, not a runtime dependency.

This is an active performance project. Portable scalar fallbacks exist, but
other CPU platforms have not received equivalent optimization or benchmarking.
The current pilots do not establish an overall win against llama.cpp.

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

## Rust API and checks

`Model::load` loads the original checkpoint; `Model::load_with_quantization`
selects an overlay and explicit `QuantizationMode`. Create a `CpuContext` with
backend, worker count and scratch budget, then call `Model::infer(&Request,
&mut CpuContext)`. The scratch limit applies to checked operator allocations;
it is not a total process-memory limit. Mapped checkpoint contents must remain
immutable throughout their lifetime.

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
final repeated benchmark and resource/API hardening remain unfinished.
