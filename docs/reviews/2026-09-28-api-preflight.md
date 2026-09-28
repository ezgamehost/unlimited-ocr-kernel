# API preflight during the resident benchmark

Runtime source inspected at `e5dd830`; benchmark tooling at `ce095b2`. This is an
implementation preflight, not the required final independent source review.
No runtime edits or heavy verification ran alongside the timed benchmark.

The following source-level gaps need regression tests and fixes after the
measurement process finishes. They do not establish failures of the valid
corpus inputs currently being timed.

1. `src/vision/spatial.rs`: public window partition/unpartition validate the
   input tensor but form derived products (`window * window`, `ny * nx`, padded
   element counts) without checked arithmetic. A tiny valid input and an extreme
   window/destination size can reach these products. Add checked derived shapes
   before allocation/indexing and regress invalid sizes without large allocations.
2. `src/preprocess.rs` and `src/vision/mod.rs`: `PreparedImage` fields are public;
   `image_tokens` and crop-grid assembly multiply caller-provided grid dimensions
   without a checked validation boundary. Normal `prepare` output is constrained,
   but the public library can receive manually constructed values. Validate the
   entire prepared image before starting vision work and make token counts checked.
3. `Vision::new` expands all matching vision tensors to FP32 and checks only the
   patch-embedding weight shape at construction. Other consumed shapes/dtypes
   are checked later, sometimes by indexing code that assumes the pinned model.
   Add a metadata-only preflight for every consumed vision tensor before expansion.
   A small checkpoint containing only the valid patch tensor is a useful negative
   fixture: it must be rejected as incomplete at construction.
4. `src/model/config.rs` validates individual nonzero dimensions, but shared-expert
   width is multiplied unchecked in `Decoder::new`. Validate derived dimensions
   as well as individual fields, including configs constructed directly in Rust.
5. Scratch checks need an aggregate-lifetime audit across image preparation,
   vision intermediates, decoder temporaries and persistent KV storage. Document
   the exact scope of `--scratch-mib`; avoid presenting a per-operation scratch
   limit as a total process-memory limit. Validate the largest admitted 32-crop
   grid with real resource evidence.

The read-only mmap file-mutation contract, newly added dynamically resolved BLAS
FFI, all SIMD preconditions, and portable fallback behavior remain in scope for
final independent review. These notes do not replace that review or its tests.

## Allocation lifetime observations

The README already correctly describes scratch as checked operator allocations,
not a process-memory ceiling. Inspection gives the following more precise scope:

- `CpuContext::check_elements` checks a supplied FP32 element count against the
  configured byte limit. It does not track outstanding allocations. Linear
  operators check their output; blocked attention checks output, a transposed
  value buffer and score blocks. Existing caller-owned inputs remain live.
- Concurrent expert execution conservatively reserves aggregate work before
  dispatch and divides the per-worker allowance. This does not include the
  whole model, prepared images or persistent decoder state.
- `LayerCache::prefill` checks capacity arithmetic and uses fallible reservation,
  independently of `CpuContext`. For the pinned 12-layer, width-1280 model its
  reserved FP32 key/value payload is
  `12 * 2 * (prefix_tokens + 128) * 1280 * 4` bytes. The matched corpus prefix
  of 1518 therefore reserves 202,260,480 bytes (about 192.89 MiB), excluding
  allocator overhead and temporary attention layouts.
- `prepare` materializes every normalized local crop. A 32-crop request retains
  `32 * 640 * 640 * 3 * 4 + 1024 * 1024 * 3 * 4 = 169,869,312` bytes
  (162 MiB) in prepared pixel tensors alone. Decoded/resized image buffers and
  preprocessing temporaries add to the peak; the 100-million-pixel input bound
  is not a byte or aggregate-memory bound.
- `Model::infer` keeps the prepared image, projected features and prompt
  embeddings in its outer scope during decoding. Explicitly releasing these
  after their final use is a concrete memory optimization to measure, with
  unchanged token-output regression coverage.
- Production `Vision::encode` calls `encode_debug`, retaining global SAM/CLIP
  diagnostic outputs while processing local views. Local crops are encoded
  serially, so their full attention intermediates do not scale by crop count.
  Separate diagnostics from production only if measured savings justify it.

These are source-derived payload calculations, not observed peak RSS. The
32-crop acceptance run and resource comparison must still measure actual child
RSS after the current timing run finishes.


## Window-shape follow-up

Item 1 is now addressed for the two public window functions: derived window
counts and output shapes use checked multiplication before indexing or allocation;
allocation uses fallible reservation; unpartition rejects empty tensor dimensions.
Regression tests reproduced an overflow panic and inconsistent empty-input
acceptance before the change. Full Rust tests and clippy pass. All six release
vision tests pass when the three original artifact-dependent fixtures are included,
including full multi-crop pixels and image embeddings. Logs are retained in
[window-bounds](window-bounds/). This is not a global scratch-memory bound and does
not resolve the remaining prepared-image/config/checkpoint descriptor findings.


## Prepared-image follow-up

Item 2 is addressed: public prepared images validate grid multiplication, the
32-tile limit, crop count, fixed global/local view shapes, and tensor storage.
Token counting is fallible (`Result<usize>`) and checks layout first. Vision
encoding obtains that validated count before any global-view work, bounding
subsequent grid assembly arithmetic. The Rust API documentation and benchmark
example are updated for the fallible count.

The new layout test fails on the missing validation before implementation,
then passes with the full Rust suite and clippy. All six release vision checks,
including original multi-crop pixel/embedding references, pass. An initial full
build caught an example call site needing the new Result handling; it was fixed
before the final checks. [Verification logs](prepared-layout/) retain the results.
Vision descriptor preflight, direct-config validation and aggregate lifetime
optimization remain open.


## Decoder-configuration follow-up

Item 4 is addressed: JSON parsing and `Decoder::new` share configuration
validation. Shared-expert width, matrix storage sizes and aggregate KV capacity
use checked arithmetic and must fit addressable FP32 storage. This is arithmetic
validation, not a promise that every accepted configuration fits physical RAM.
Decoder configuration is private after construction, with read-only access via
`config()`, so callers cannot invalidate checked descriptors by mutating it.

Regressions reject overflowing JSON dimensions and directly modified Rust
configurations before checkpoint descriptor access. The full Rust suite, clippy,
and both original decoder reference tests pass. [Logs](derived-config/) retain
RED/GREEN and artifact-dependent evidence. Vision descriptor validation and
allocation-lifetime optimization remain open.
