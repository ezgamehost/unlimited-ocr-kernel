# Hybrid quantization fails the 32-crop stress case

The hybrid Q4-body/Q8-head decoder with dynamic INT8 activations is not a
general accuracy replacement for BF16 or Q8. On the frozen synthetic 640×20,480
image it generates unrelated Chinese text repeatedly among the English sections.
Character error rate is 1.3664289632, compared with 0.0691168404 for current
BF16 and dynamic Q8. CER can exceed one because insertions count as errors.

| Configuration | Emitted tokens | Finish | CER |
|---|---:|---|---:|
| Hybrid, token-major cache | 3349 | EOS | 1.366429 |
| Hybrid, head-major cache | 3349 | EOS | 1.366429 |
| Current BF16 weights / FP32 compute | 930 | EOS | 0.069117 |
| Current Q8 weights / dynamic INT8 activations | 930 | EOS | 0.069117 |

The hybrid outputs match exactly across cache layouts, including text, token
IDs, EOS and 3,793 image tokens. This extends cache-layout validation to a
larger prefix but also exposes a quantization-quality gap missed by the twelve
smaller prompt/page cases. It is a failed quality case, not a speed benchmark.

Both current BF16 and Q8 outputs also match their earlier original-prompt
controls exactly, including token IDs. Thus the recent vision/cache changes do
not explain this observed regression. The hybrid precision path is implicated;
additional controls isolate activation precision and packed storage. Until those
finish, do not attribute the failure more narrowly to either component.

The BF16/Q8 result is itself imperfect: it omits sections 01–02 and repeats
section 26. Original-upstream attribution of those remaining errors is still
pending. This extreme aspect-ratio image is a crop-limit stress test, not a
representative document corpus. It remains evidence against generalizing the
four-page hybrid speedup to unrestricted OCR accuracy.

All runs use the original native prompt `<image>document parsing.`, an
8,192-token cap, eight Rust workers and eight-thread BLAS environment limits.
The before/after-cache hybrid binaries are pinned by hash in `run.py`; the
precision controls use the current head-major binary, SHA
`c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57`.
The image SHA is
`4fe946abc026998f6e1a06f878fd7a309aae1d8785fea5d51cc6a25d4a63d01e`.
Raw outputs, normalized transcripts, command identities and process resources
are preserved. These are single correctness/resource observations on battery;
their elapsed times are not comparative performance results.

Reproduce `run.py` from the repository root with `PYTHONPATH=.` and fresh
output directories. The BF16/Q8 control uses the sibling
`2026-09-28-receipt-diagnostic/native_default.py` with `--image` and `--truth`
pointing to `2026-09-28-max32/portrait32.png` and `.txt`, and `--output` pointing
to a new control directory. `weight_only.py` changes only activation precision;
its `--mode simulation` instead retains dynamic activations and uses the
previously audited physical-INT8 simulation of the same hybrid codes/scales.
