# Four-row SDOT decoder kernel

This candidate shares each 32-byte activation block across four row-major
weight rows. Four independent integer block sums are reduced into four FP32
lanes, then scaled and accumulated in the original order, without FMA.
Weights are not repacked and no additional model-sized copy is allocated.
Incomplete row groups and non-block-aligned columns use the previous kernel.

The comparison base is commit `48ac31b`. `kernel.patch` records the candidate.
The M4 Pro runs eight CPU workers, original BF16 vision and the validated
block-32 INT8 decoder overlay with dynamic INT8 activations. The model and
overlay are the same pinned artifacts documented in the preceding report.

## Fixed-work native decoder measurement

The ignored `model::benchmark::fixed_work_decoder_benchmark` unit benchmark
loads the model once, injects the original English reference image embeddings,
and creates a fresh prefix cache for every repetition. The prefix contains
1,517 tokens, including 1,513 image tokens. Each repetition executes exactly
512 decoder steps using the same 250 recorded English output tokens cyclically.
EOS is deliberately ignored. This is a decoder workload, not completed OCR and
not a direct llama.cpp comparison. Timing includes a checksum over all logits.
Vision and model loading are excluded; prefill is reported separately.

Two release test executables were built from the same harness, one with the
base kernel and one with the candidate. They run sequentially in base,
candidate, candidate, base order. Each process performs one warmup and three
measured repetitions. No concurrent compilation or other benchmark was run.
The complete process logs retain warmups, checksums, and every timing.

Reproduce the candidate with:

```sh
RUSTUP_HOME="$PWD/.tools/rustup" CARGO_HOME="$PWD/.tools/cargo" \
  .tools/cargo/bin/cargo test --release --lib fixed_work_decoder_benchmark \
  -- --ignored --nocapture --test-threads=1
```

## Full OCR and operator checks

`fresh/` preserves four fresh-process runs, commands, resource logs and output.
All four reach EOS with exactly the prior native token IDs. CER is zero for
English, Chinese and dense text; the table retains its existing one-character
error (CER 0.003984). Decode times are mixed on short pages. Dense decode drops
from 16.642 to 14.873 seconds in this single pilot. This is not a repeated
end-to-end speedup claim against llama.cpp.

The hot expert projection improves in the operator pilot, while the vocabulary
head shows no improvement. The direct four-row test checks bitwise equality to
the previous single-row kernel, including negative integer extremes and varied
scales. Whole-model fixed-work checksums complement the unchanged OCR outputs.

Fixed-work results: baseline median 5.136983 seconds, candidate median 4.580291
seconds (1.122x throughput, 10.8% less elapsed time), six measured samples
per version. All 16 repetitions, including warmups, share the same logits
checksum. This supports retaining the kernel; it does not establish a
completed OCR win over llama.cpp.

Verification: full Rust suite 41 passed, eight artifact-dependent tests
ignored by default; release unit/quantized suite seven passed, benchmark
ignored by default; clippy all targets with warnings denied passed. The
artifact-dependent fixed-work benchmark was executed explicitly as above.
