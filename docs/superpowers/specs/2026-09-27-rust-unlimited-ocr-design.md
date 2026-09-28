# Unlimited-OCR Rust CPU runtime

Status: design direction approved on 2026-09-27; llama.cpp CPU smoke benchmark
completed. Rust implementation and representative document benchmarks remain.

## Intent and scope

Build a standalone Rust CPU inference runtime for baidu/Unlimited-OCR,
initially optimized and measured on the local Apple M4 Pro with 24 GiB RAM.
The user approved Rust, M4 Pro first, and Python for weight preparation and
reference generation, and requested an initial vLLM benchmark.
Inference itself must not require Python. Success means real image-to-text
inference, validated against upstream, followed by measured performance work.
Microbenchmarks alone do not satisfy this objective.

First deliverable: a local llama.cpp CPU baseline and reference fixtures;
vLLM measurement remains dependent on access to supported hardware.
Second: correct single-image Rust inference, including base and gundam modes.
Third: measured CPU optimization. Multi-page support and GPU backends are
subsequent extensions; interfaces should allow them without implementing them now.

## Evidence and model identity

Pin model source, tokenizer, configuration and weights to Hugging Face revision
`07dea832e22aefee32ad281d4b80551282e1c168`. Record hashes of downloaded artifacts.

The configuration specifies hidden size 1280, 12 decoder layers, 10 attention
heads and 10 KV heads, head dimension 128, vocabulary 129280, 64 routed experts,
six selected experts per token, two shared experts, and one initial dense layer.
The vision pipeline uses SAM and CLIP followed by a 2048-to-1280 projector.
These are implementation inputs, not a substitute for auditing upstream forward
passes, configuration defaults and checkpoint tensor shapes.

The upstream attention implementation preserves all prefill KV entries and
adds a 128-entry rolling decode region. Implement full causal prefill, then
single-token decode with absolute rotary positions and in-place ring updates.
Do not discard the image/prompt prefix or treat this as an ordinary 128-token
sliding window. Generation repetition suppression has a separate window.

Sources:
- https://huggingface.co/baidu/Unlimited-OCR/tree/07dea832e22aefee32ad281d4b80551282e1c168
- https://huggingface.co/baidu/Unlimited-OCR/blob/07dea832e22aefee32ad281d4b80551282e1c168/modeling_deepseekv2.py
- https://recipes.vllm.ai/baidu/Unlimited-OCR

## Architecture and alternatives

Use a custom Rust runtime with narrowly scoped native acceleration. This gives
control over model-specific attention, expert weight access and memory use.
Extending an existing Rust tensor engine would reduce basic operator work but
still requires a model port and may constrain cache and expert execution.
A C runtime with Rust bindings adds language boundaries without an identified
performance benefit for this project. Choose custom Rust, with measured reuse
of Accelerate through its C interface where beneficial.

Separate modules own checkpoint validation/loading, tensor storage, CPU kernels,
image processing, vision model, decoder/cache, generation, and CLI reporting.
Use an explicit CPU execution context to select scalar, NEON, or Accelerate
operations. Avoid designing a general graph compiler or GPU scheduler now.

Memory-map read-only safetensors, validate dtype/shape/offsets before exposing
typed views, and keep the mapping alive for every view. Initially retain BF16
weight storage with FP32 computation; convert or pack bounded working sets.
Do not expand the complete checkpoint to FP32 plus a second packed copy.
Plan scratch allocations before execution and report estimated memory needs.
Use blocked attention to avoid materializing unbounded attention-score matrices.
Return clear errors for unsupported configurations, corrupt files, allocation
limits and invalid image inputs.

Safe Rust owns tensors, mappings, threads and caches. Isolate unsafe SIMD and
FFI behind checked dimensions, explicit alignment rules and documented safety
invariants. Keep portable scalar implementations as numerical references.
Use one bounded threading policy and prevent nested BLAS/thread-pool oversubscription.

Port image resizing, padding, cropping, normalization, patch ordering, image
special tokens and prompt construction from pinned upstream code. Reuse a Rust
tokenizer implementation compatible with the checkpoint tokenizer JSON, tested
against upstream IDs. Preserve structured special tokens in raw output.
Provide CLI inputs for weights, image, image mode, backend, thread count and
output limit; emit text and machine-readable timing/error reports.

## Initial vLLM benchmark

The requested vLLM baseline is hardware-blocked; the user accepted the local
llama.cpp smoke benchmark and requested aggressive Rust CPU optimization.
Continue implementation without waiting for vLLM hardware. The official recipe uses a
dedicated CUDA Docker image and an NVIDIA GPU. The local host is macOS ARM64;
the user confirmed that only the M4 Pro is available.
Do not claim a run, fabricate timing, or substitute another inference engine.
If only the Mac is available, record the baseline as blocked and investigate
model-specific CPU support separately; generic vLLM CPU support is insufficient.

The user subsequently proposed llama.cpp. Use release b11223, commit
`4da6337767f973e2b4d0797e5b323d77d8565e4a`, for an initial local CPU smoke
benchmark with GPU layer, operation, KV and projector offloading disabled.
The current converter documents that its OCR decoder ignores sliding-window
metadata and uses full attention. This makes it a useful practical CPU baseline,
but not a reference for the original model's long-generation attention behavior.
Record precision and sampling differences explicitly. Start with community Q8_0
language weights and F16 vision weights from
`vimalnakrani/unlimited-ocr-gguf`, revision
`45cd66ec6b46a7c4de49f376084ecec2b8d3c59a`.
The upstream baidu.png is only a logo smoke test, not a document benchmark.

Use the official model recipe, including its n-gram logits processor, literal
`<image>document parsing.` prompt, special-token retention, and disabled prefix
and multimodal processor caches. Pin the actual container digest and model
revision at execution, and save the complete server and request configuration.
Use temperature zero, ngram size 35, repetition window 128, and 8192 maximum
output tokens for the initial single-image gundam baseline. Any truncation is
reported and excluded from claims of completed page OCR. Verify preprocessing
mode and effective token limit on both runtimes before comparison.

Start with the upstream baidu.png image as a smoke fixture, then a versioned
corpus of English, Chinese, table, and dense-document images with transcriptions
for accuracy evaluation. A single smoke image is not a representative workload.
Hash all inputs. Use concurrency one, two warmups per image, then ten measured
runs per image; retain individual records and report median and p95 latency.
Report cold model startup separately. Add concurrency sweeps only after this
latency baseline is stable.

Collect client end-to-end latency, time to first content, completion token count,
finish reason, full output, errors and available server timing/memory metrics.
Streaming chunks are not tokens: calculate decode throughput only from valid
token counts and decode timing; mark unavailable metrics explicitly. Record
network overhead as part of remote client latency. Measure peak server CPU/GPU
memory at the server, never substitute client memory usage. Include hardware,
OS, software versions, dtype, threads, cache policy and model revision.

A GPU vLLM result is a system-level reference, not evidence of CPU kernel speedup.
Measure scalar Rust versus optimized Rust on the same M4 Pro as the CPU speedup
comparison. Keep quantized and unquantized results separate.

## Correctness and performance gates

Generate upstream fixtures for preprocessing, image embeddings, selected decoder
layers, router decisions, logits, and teacher-forced decode beyond two complete
ring wraps. Record reference device/dtype and source revision. Use deterministic
small FP32 operator fixtures with atol 1e-5 and rtol 1e-4; report maximum absolute
and relative errors. For full BF16-model comparisons, report discrepancies at
each boundary and establish justified tolerances from reference precision
experiments before declaring parity. Do not silently relax failing tolerances.

Test invalid tensor shapes, malformed/truncated weights, odd SIMD dimensions,
nonfinite normalization inputs, ring-wrap boundaries, EOS, token limits, image
aspect ratios, crop ordering and tokenizer special tokens. Require matching
token IDs for preprocessing/tokenization fixtures. Compare teacher-forced logits
to distinguish numerical drift from autoregressive divergence; compare final
OCR against ground truth using character error rate and retain raw output.

Optimize only after real inference passes its reference checks: profile image
encoding, prefill, expert execution, output projection and decode separately.
Benchmark Accelerate GEMM against blocked Rust/NEON; target expert GEMV and
memory traffic where profiling warrants it. Reuse scratch and cache storage.
Then evaluate weight-only INT8, followed by INT4 only with an accuracy report.
No fixed speedup or throughput is promised before measurement.

## Completion evidence

Deliver a buildable Rust library and CLI, pinned preparation/reference tools,
operator and model checks, reproducible benchmark commands, raw measurements,
and a report stating what ran and any missing hardware-dependent checks.
vLLM measurement remains an explicit outstanding requirement until a real run
completes. Rust inference completion requires actual checkpoint/image execution,
not stubs, synthetic-only tests, or kernel microbenchmarks.
