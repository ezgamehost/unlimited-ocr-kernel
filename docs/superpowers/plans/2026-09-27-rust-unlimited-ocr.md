# Rust Unlimited-OCR CPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correct native Rust Unlimited-OCR runtime, then reduce measured M4 Pro CPU document latency relative to llama.cpp without hiding accuracy regressions.

**Architecture:** Memory-mapped BF16 checkpoint storage feeds checked Rust tensor operations, SAM/CLIP vision, and a model-specific MoE decoder. Scalar kernels establish correctness; Accelerate and isolated NEON kernels optimize measured bottlenecks. Python prepares reference fixtures and benchmarks, never serving as the inference runtime.

**Tech Stack:** Stable Rust, Cargo, safetensors, memmap2, half, tokenizers, image, serde/serde_json, clap, rayon, Criterion, Apple Accelerate CBLAS; Python with PyTorch/Pillow for references. Pin resolved versions and toolchains during setup; disable GPU dependencies.

**Spec:** `docs/superpowers/specs/2026-09-27-rust-unlimited-ocr-design.md`

## Global Constraints

- Primary target: Apple M4 Pro CPU, 24 GiB RAM; portable scalar fallback.
- Model revision: `07dea832e22aefee32ad281d4b80551282e1c168`.
- Inference itself must not require Python.
- Safe Rust owns tensors, mappings, threads and caches.
- Do not expand the complete checkpoint to FP32 plus a second packed copy.
- Use one bounded threading policy and prevent nested BLAS/thread-pool oversubscription.
- FP32 operator fixture tolerances: atol 1e-5 and rtol 1e-4.
- Never silently relax failing tolerances; establish model tolerances from recorded reference-precision experiments.
- Baseline: llama.cpp b11223 / `4da6337767f973e2b4d0797e5b323d77d8565e4a`, CPU only.
- Q8 baseline repo/revision: `vimalnakrani/unlimited-ocr-gguf` / `45cd66ec6b46a7c4de49f376084ecec2b8d3c59a`.
- Original semantics: full causal prefill plus preserved prefix and 128-entry generated-token ring, absolute rotary positions.
- Actual checkpoint/image execution is required for completion; synthetic-only tests and microbenchmarks are insufficient.
- vLLM remains unmeasured because only the Mac is available; this does not block the authorized Rust work.

## Review Focus

- Truncated/overflowing checkpoint offsets must return errors before unsafe access (Task 2).
- Repeated requests must not reuse another image's embeddings, KV state or n-gram history (Tasks 4 and 6).
- Extremely wide/tall images must preserve upstream crop order and respect bounded scratch allocation (Tasks 3 and 5).
- Near-tied expert routing and logits must expose numerical divergence rather than claim exact parity (Tasks 1 and 4).
- Repetition or early EOS must not appear as a latency win through shorter, worse output (Tasks 6 and 8).

## File and Interface Map

One library crate `uocr` and binary `uocr`; modules below are separate files/directories.
Public interfaces return `Result<T, OcrError>` unless otherwise stated. Use a single
error type with I/O, format, shape, unsupported-configuration, numerical and resource-limit variants.

- `src/error.rs`: error type and contextual messages.
- `src/checkpoint.rs`: `Checkpoint::open(path: &Path) -> Result<Checkpoint>` and `tensor(&self, name: &str) -> Result<WeightView<'_>>`; views borrow the mapping.
- `src/tensor.rs`: `Tensor { shape: Vec<usize>, data: Vec<f32> }`, checked constructors and `WeightView<'a>` with validated shape/dtype/byte view.
- `src/cpu/mod.rs`: `CpuContext::new(backend: Backend, threads: usize, scratch_limit: usize) -> Result<CpuContext>`; `Backend::{Scalar, Neon, Accelerate, Auto}`. Owns bounded worker/scratch resources.
- `src/cpu/ops.rs`: checked linear, convolution, normalization, activation and attention entry points, consuming tensor/weight views plus `&mut CpuContext`.
- `src/model/config.rs`: `ModelConfig::load(path: &Path) -> Result<ModelConfig>`; validates supported architecture and resolves upstream defaults explicitly.
- `src/preprocess.rs`: `prepare(image: &Path, mode: ImageMode) -> Result<PreparedImage>`; `ImageMode::{Base,Gundam}`, global/crop tensors and grid metadata.
- `src/prompt.rs`: `PromptEncoder::load(path: &Path) -> Result<PromptEncoder>`; `encode(&self, prompt: &str, image_tokens: usize) -> Result<PromptInput>`; IDs and image insertion positions.
- `src/vision/mod.rs`: `Vision::encode(&self, image: &PreparedImage, cpu: &mut CpuContext) -> Result<Tensor>`; final ordered image embeddings.
- `src/decoder/mod.rs`: `Decoder::prefill(&self, embeddings: &Tensor, cpu: &mut CpuContext) -> Result<(DecodeState, Tensor)>`; `step(&self, token: u32, state: &mut DecodeState, cpu: &mut CpuContext) -> Result<Tensor>` returns next-token logits.
- `src/generation.rs`: `GenerationOptions { max_tokens: usize, ngram_size: usize, ngram_window: usize }`, `FinishReason::{Eos,TokenLimit}`, raw output IDs and timings.
- `src/model/mod.rs`: `Model::load(dir: &Path) -> Result<Model>`; `infer(&self, request: &Request, cpu: &mut CpuContext) -> Result<OcrResult>`. Model owns checkpoint; layer descriptors store names/offsets, avoiding self-referential borrowed structs.
- `src/bin/uocr.rs`: one-shot and sequential JSONL request modes using the same model API.
- `tools/`: pinned model acquisition, upstream-reference adapters, corpus generation, benchmark and scoring utilities.
- `tests/fixtures/`: small deterministic numerical fixtures; full weights and large tensors stay in ignored artifact directories.

## Task 1: Reproducible references and meaningful baseline

**Files:** `tools/fetch_model.py`, `tools/reference.py`, `tools/corpus.py`, `tools/benchmark.py`, `tools/score.py`, `tests/python/test_benchmark.py`, `benchmarks/corpus/manifest.json`, `requirements-reference.txt`, `.gitignore`.
**Consumes:** pinned upstream source and existing `docs/benchmarks/2026-09-27-llama-cpu` evidence.
**Produces:** hashed checkpoint manifest, reference fixture manifest, versioned document corpus, per-request JSON benchmark records.

- [ ] Write tests proving failed requests cannot enter successful latency aggregates; token-limit outputs are marked incomplete; identical strings score CER=0 and one substitution in four characters scores CER=0.25; image hashes must match.
- [ ] Run `python3 -m unittest discover -s tests/python` and confirm the new cases fail before implementation.
- [ ] Implement artifact download with pinned URLs/hashes and atomic completion. Prepare English prose, Chinese text, table and dense-text images with exact transcriptions and recorded fonts/layout; retain the logo only as smoke input. Validate Chinese font coverage before rendering.
- [ ] Build a pinned upstream CPU adapter: invoke model components without high-level CUDA-only helpers; record every device/dtype adaptation, preserve tensor math. Generate preprocessing, SAM/CLIP, decoder/router/logit and 300-step teacher-forced fixtures. Compare CPU FP32 and supported CPU BF16 paths to establish error envelopes; if BF16 execution is unsupported, mark that reference unavailable, never emulate it silently.
- [ ] Implement independent-process and persistent-model measurements as separately labeled modes. Use 2 warmups and 10 repetitions per image, concurrency 1, record raw output, token count, finish reason, wall time and server/process memory. Use temperature 0 and cap 8192 for document runs. Do not equate streaming chunks with tokens.
- [ ] Rerun tests and baseline on the corpus. Record llama.cpp full-attention and repetition-processor differences. Pin a BF16 GGUF baseline as well as existing Q8 for precision-aware comparison. Commit tools, small fixtures and metadata, not weights.

## Task 2: Checked storage and CPU execution foundation

**Files:** `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `src/lib.rs`, `src/error.rs`, `src/checkpoint.rs`, `src/tensor.rs`, `src/cpu/{mod,ops,scalar,accelerate}.rs`, `src/model/config.rs`, `tests/{checkpoint,ops}.rs`.
**Consumes:** checkpoint/fixture manifests from Task 1.
**Produces:** validated weight views, model configuration, scalar operators and checked Accelerate GEMM dispatch.

- [ ] Install an isolated Rust toolchain if needed and record its version; add fixtures/tests for malformed headers, overlapping/out-of-bounds offsets, multiplication overflow, wrong dtypes, shape mismatch and mmap view lifetimes. Add FP32 reference tests for matrix products, conv2d, stable softmax, LayerNorm/RMSNorm, GELU/SiLU and RoPE, including odd widths.
- [ ] Run `cargo test --test checkpoint --test ops` and confirm failure before implementation.
- [ ] Implement the mapped loader and checked operators. Validate file length before mapping, disallow concurrent mutation of active weight files, use checked byte decoding instead of assuming BF16 alignment. Audit and document any unsafe blocks.
- [ ] Add macOS Accelerate FFI for dense FP32 GEMM, with explicit row-major layouts and checked leading dimensions. Keep source/converted tiles bounded. Select scalar fallback elsewhere. Pin one threading policy per operation; do not nest Rayon workers around internally threaded BLAS.
- [ ] Run operator fixtures at atol 1e-5/rtol 1e-4 and `cargo clippy --all-targets -- -D warnings`; load the actual checkpoint and validate every consumed tensor descriptor. Commit passing implementation.

## Task 3: Exact image preparation and prompts

**Files:** `src/preprocess.rs`, `src/prompt.rs`, `tests/{preprocess,prompt}.rs`.
**Consumes:** Task 1 upstream fixtures and Task 2 tensors/config.
**Produces:** `PreparedImage` and `PromptInput` for both image modes.

- [ ] Add tests for square, narrow, wide, small, alpha-bearing and grayscale images, crop-grid tie cases and token insertion positions; require exact tokenizer IDs and crop order versus fixtures.
- [ ] Run `cargo test --test preprocess --test prompt` to establish failures.
- [ ] Implement upstream resize/padding/normalization semantics and grid selection. Verify interpolation numerics against Pillow fixtures rather than assuming the Rust image library's named filter is equivalent. Bound dimensions and crop counts before allocating.
- [ ] Load the checkpoint tokenizer and match literal prompt/special-token behavior; keep image positions separate from ordinary text IDs.
- [ ] Pass fixture checks and malformed-image errors; commit.

## Task 4: Decoder and preserved-prefix attention

**Files:** `src/decoder/{mod,attention,cache,moe}.rs`, `tests/{decoder,cache,moe}.rs`.
**Consumes:** Task 2 storage/ops and Task 1 decoder fixtures.
**Produces:** decoder prefill/step APIs and independently resettable `DecodeState`.

- [ ] Add tests for dense initial layer, shared/routed expert mixing and upstream routing normalization/tie behavior. Test prefix preservation at generated lengths 127, 128, 129, 255, 256 and 300; compare against a simple explicit attention reference with absolute rotary positions.
- [ ] Run decoder/cache/MoE tests and confirm failures.
- [ ] Implement decoder layers with full causal prefill, blockwise attention and per-layer prefix-plus-ring KV caches. Compute only selected experts. Stream logits without duplicating the vocabulary projection weights. Keep state per request and reject inconsistent positions.
- [ ] Compare intermediate states and teacher-forced logits against Task 1 fixtures; report error maxima and routing differences. Diagnose divergent near-ties before setting acceptance thresholds.
- [ ] Verify reset isolation across two different requests and EOS/token bounds; commit.

## Task 5: SAM/CLIP vision and multimodal assembly

**Files:** `src/vision/{mod,sam,clip,projector}.rs`, `tests/vision.rs`.
**Consumes:** Task 3 `PreparedImage`, Task 2 operators and Task 1 vision fixtures.
**Produces:** final ordered image embeddings for base and gundam.

- [ ] Add per-stage tests for patch convolution, SAM window partition/padding, global attention with relative positions, neck/downsampling, CLIP positional interpolation/class token, linear projector and newline/view separator insertion.
- [ ] Run `cargo test --test vision` and establish failures.
- [ ] Implement SAM and CLIP exactly from pinned forward passes; feed SAM features into CLIP as upstream does. Concatenate CLIP patch outputs with SAM features before projection. For cropped images, preserve the source implementation's local/global/view-separator order, not assumptions from configuration field names.
- [ ] Reuse bounded scratch across blocks and process crops within the memory budget; avoid full global score matrices. Check projected token count against prompt image positions before scatter.
- [ ] Compare real-image intermediate fixtures in both modes; verify memory estimates on the largest admitted crop grid; commit.

## Task 6: Complete Rust OCR API and CLI

**Files:** `src/model/mod.rs`, `src/generation.rs`, `src/bin/uocr.rs`, `tests/{generation,inference}.rs`, `README.md`.
**Consumes:** Tasks 2–5 APIs.
**Produces:** Python-free one-shot inference and sequential JSONL requests with stage timing.

- [ ] Add tests for upstream windowed 35-gram suppression, EOS, exact token caps, raw special-token retention and request isolation; add integration fixtures for real checkpoint/image execution, opt-in when large artifacts are absent.
- [ ] Run generation/inference tests and confirm failures.
- [ ] Compose preprocessing, embeddings, prefill and greedy generation. Apply the original n-gram suppression semantics, preserve raw output and record finish reason. Reset all per-request state in JSONL mode; keep only immutable weights/packed data across requests.
- [ ] Implement flags for model directory, image, mode, CPU backend, threads, memory limit and token cap; send machine-readable timing separately from OCR text. Report corrupt/unsupported artifacts without panics.
- [ ] Run actual single-image inference in both modes, then the full corpus. Check upstream teacher-forced logits and OCR CER; investigate discrepancies before optimization. Commit a functional baseline.

## Task 7: Profile and optimize hot kernels

**Files:** `src/cpu/{neon,packed,dispatch}.rs`, targeted changes in CPU/vision/decoder modules, `benches/{linear,attention,moe,vision}.rs`, `tests/optimized.rs`.
**Consumes:** verified Rust baseline and stage profiles.
**Produces:** checked optimized backend dispatch with scalar equivalence and measured per-stage results.

- [ ] Profile real image encoding, prefill, expert GEMV and vocabulary projection; record allocations, conversion/packing cost and time per stage. Establish meaningful layer-shaped benchmarks, including cold and reused packed weights.
- [ ] Write differential tests before each optimized path, including unaligned inputs, tails, extreme values and multiple thread counts; confirm missing-backend tests fail.
- [ ] Implement NEON blocked BF16-to-FP32 GEMV for decode and benchmark Accelerate GEMM for vision/prefill. Fuse norm/activation/residual work only when it eliminates measured memory traffic. Preserve portable scalar fallback and checked dispatch.
- [ ] Prepack immutable weights once when repeated-use gains justify memory; cap packed-cache size and account for eviction cost. Batch selected experts during prefill; avoid computing unused experts during decode.
- [ ] Tune tile dimensions and thread counts 1/4/8/10/14 on held-out benchmark images. Keep only changes that improve end-to-end latency without reference failures. Run `cargo test`, release integration tests, formatting and clippy; commit each accepted optimization with measurements.

## Task 8: Quantization and fair final comparison

**Files:** `tools/quantize.py`, `src/cpu/quantized.rs`, `tests/quantized.rs`, `docs/benchmarks/rust-cpu/`, documentation updates.
**Consumes:** correct optimized BF16-storage runtime and versioned corpus.
**Produces:** optional INT8 path, accuracy/performance report, reproducible commands and final raw benchmark records.

- [ ] Add roundtrip/scale/zero-range/tail tests and quantized dot-product reference tests before implementing the packed format; confirm initial failures.
- [ ] Implement versioned weight-only INT8 preparation with explicit scales and metadata, then its NEON kernel. Keep normalization/routing tensors in reference precision. Validate checksum, dtype and layout before dispatch.
- [ ] Measure CER and latency versus unquantized Rust and llama.cpp Q8. Accept quantization as an optional speed/size trade only with its measured error report; do not label it equivalent based solely on a logo. Attempt INT4 only if INT8 results and profiling justify further accuracy experiments.
- [ ] Run 2 warmups and 10 measured repetitions per corpus image for each configuration. Compare matching process-lifetime modes, threads, input modes and caps; retain incomplete outputs, failures and per-image CER. Add long outputs past the 128-token ring boundary and disclose attention-policy differences.
- [ ] Report median/p95 total latency, first-token latency, sustained decode throughput, peak RSS, load/packing time and OCR accuracy. A speedup claim requires completed comparable OCR output; shorter/truncated output is not a win. No fixed speedup is promised.
- [ ] Complete final source/unsafe review and required tests; commit results and reproduction instructions. Explicitly identify unavailable vLLM/original-GPU checks and any unmet model-parity gates.

## Execution Recommendation

Use native execution in this session because preprocessing, vision, decoder,
and reference work depend closely on each other's interfaces. Complete a final
independent review once the full implementation works. Keep experimental kernel
variants behind explicit backend selection until they pass equivalence checks.

The existing 4.545-second logo measurement is an initial checkpoint, not the
acceptance target for all documents. The primary win is lower completed-page
latency at comparable OCR quality on the versioned corpus. Report any losses.
