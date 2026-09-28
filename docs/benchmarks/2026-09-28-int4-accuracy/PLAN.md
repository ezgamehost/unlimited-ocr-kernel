# INT4 accuracy gate before packed-kernel work

The accepted runtime remains unchanged. The first experiment quantizes supported
decoder matrices to symmetric signed values -7..7 with block32 FP32 scales, but
stores those values in the existing I8 overlay format. This permits the existing
validated W8A8 path to execute the same numerical operation as a future packed
W4A8 kernel. Routers, embeddings, norms and vision retain original precision.

This is an accuracy simulation, not 4-bit storage or a speed experiment. Overlay
metadata and sidecar explicitly declare simulated_weight_bits=4, storage_bits=8.
Runtime result precision labels describe its I8 storage/activation path; the
experiment report must additionally carry the simulated weight precision.
The source checkpoint is immutable and the usual source/output checksum gates
remain enabled. Default 8-bit exports keep their existing encoding and metadata.

1. Export candidate from the original checkpoint and retain its manifest/hash.
2. Run baseline Q8 and simulated Q4 on all eight versioned synthetic pages using
   the original native prompt, identical greedy/ngram settings, cap8192, eight
   workers/BLAS threads and no cross-request KV reuse. Preserve every response.
3. Compare completion, text CER, layout/raw token differences and empty outputs.
   Reject broad adoption if the candidate worsens quality; a narrower precision
   allocation may be investigated with an explicit new experiment.
4. Only if accuracy is viable, implement packed storage and SIMD/fallback kernels.
   Require numerical parity with the simulation before measuring performance.
   Full-page timing and memory, not simulation timing, decide acceptance.

No runtime speedup, memory saving or model-quality acceptance is claimed yet.
