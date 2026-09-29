# Independent review

Reviewer fp64_oracle_review found no material graph correctness issue for this
fixed prefill fixture. The reviewer checked RMSNorm, RoPE, causal attention,
routing/grouping/restoration, shared experts, residuals and output head against
the original source/config, and independently recomputed stored artifact metrics.
They did not rerun inference or hash the full checkpoint.

The review confirmed the narrow interpretation: native is farther from the
widened mathematical reference on this fixture. This does not identify a faulty
operation or establish behavior across inputs, decode, or vision. Expert-set
comparisons are upstream FP32 versus FP64, not native routing or ordering. Exact
FP32 validation covers graph outputs, not token/position metadata.

The reviewer noted missing standalone output/source/environment provenance.
provenance.json now records the FP64 artifact, upstream sources/config/manifest,
exporter hashes, Python/Torch/Transformers versions, platform and Torch build.
This supplements the model/input/native/script hashes already in results.json.
